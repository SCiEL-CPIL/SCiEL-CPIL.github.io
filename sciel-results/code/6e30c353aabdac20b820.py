import time
import numpy as np
from itertools import product

def construct():
    """
    Minimize the number of monochromatic 4-term arithmetic progressions
    in a 2-coloring of [1, 256].
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 256
    K = 4  # Length of arithmetic progression
    
    # Precompute all 4-term APs
    # A 4-AP is (a, a+d, a+2d, a+3d) where 1 <= a and a+3d <= 256
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Precompute a matrix representation for fast evaluation
    # For each AP, we need to check if all 4 positions have the same color
    # Let x be a binary vector of length N (0-indexed)
    # For an AP (i, j, k, l), it's monochromatic if x[i]==x[j]==x[k]==x[l]
    # This equals 1 if x[i]*x[j]*x[k]*x[l] == 1 OR (1-x[i])*(1-x[j])*(1-x[k])*(1-x[l]) == 1
    
    # We can vectorize this: for each AP, compute the product of the 4 bits
    # and the product of the 4 (1-bit) values, then sum
    
    # Create an array of indices for all APs
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    def count_mono(x):
        """Count monochromatic 4-APs for binary vector x (0-indexed, length N)."""
        # x is a numpy array of 0s and 1s
        # For each AP, get the 4 values
        vals = x[ap_indices]  # shape (num_aps, 4)
        # Product of the 4 bits
        prod1 = np.prod(vals, axis=1)  # 1 if all are 1
        # Product of (1 - bit)
        prod0 = np.prod(1 - vals, axis=1)  # 1 if all are 0
        return int(np.sum(prod1 + prod0))
    
    # Thue-Morse initialization
    def thue_morse(n):
        """Return the Thue-Morse sequence up to length n."""
        tm = [0] * n
        for i in range(n):
            tm[i] = bin(i).count('1') % 2
        return np.array(tm, dtype=np.float64)
    
    # Best solution found
    best_x = thue_morse(N)
    best_score = count_mono(best_x)
    
    # Simulated Annealing
    def simulated_annealing(x, max_iter, start_temp, end_temp, deadline):
        """Run simulated annealing on binary vector x."""
        x = x.copy()
        current_score = count_mono(x)
        best_local_x = x.copy()
        best_local_score = current_score
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Choose a random position to flip
            pos = np.random.randint(N)
            x[pos] = 1 - x[pos]
            
            new_score = count_mono(x)
            delta = new_score - current_score
            
            # Temperature schedule
            t = start_temp * (end_temp / start_temp) ** (i / max_iter)
            
            if delta < 0 or np.random.random() < np.exp(-delta / t):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_x = x.copy()
            else:
                # Undo the flip
                x[pos] = 1 - x[pos]
        
        return best_local_x, best_local_score
    
    # Multiple restarts with different initializations
    initializations = [
        thue_morse(N),
        np.random.randint(0, 2, N).astype(np.float64),
        np.random.randint(0, 2, N).astype(np.float64),
        np.random.randint(0, 2, N).astype(np.float64),
    ]
    
    for init_idx, init_x in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Run SA from this initialization
        max_iter = 50000
        start_temp = 1.0
        end_temp = 0.01
        local_x, local_score = simulated_annealing(init_x, max_iter, start_temp, end_temp, deadline)
        
        if local_score < best_score:
            best_score = local_score
            best_x = local_x
    
    # Try some structured colorings as well
    structured = []
    
    # Block coloring with different block sizes
    for block_size in [2, 3, 4, 5, 6, 7, 8]:
        x = np.zeros(N, dtype=np.float64)
        for i in range(N):
            x[i] = (i // block_size) % 2
        structured.append(x)
    
    # Modular coloring
    for mod in [3, 4, 5, 6, 7, 8]:
        x = np.zeros(N, dtype=np.float64)
        for i in range(N):
            x[i] = (i % mod) < mod // 2
        structured.append(x)
    
    for x in structured:
        score = count_mono(x)
        if score < best_score:
            best_score = score
            best_x = x
    
    # Final local search on the best solution found
    if time.time() < deadline:
        # Greedy local search: flip bits if it improves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for pos in range(N):
                if time.time() > deadline:
                    break
                old_val = best_x[pos]
                best_x[pos] = 1 - old_val
                new_score = count_mono(best_x)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_x[pos] = old_val
    
    return [int(v) for v in best_x]

# Set random seed for reproducibility
np.random.seed(42)
