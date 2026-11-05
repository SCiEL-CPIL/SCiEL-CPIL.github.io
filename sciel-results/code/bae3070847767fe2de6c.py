import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 24
    # Use a high resolution for sup norm approximation
    # 4096 points is sufficient for N=24 to capture peaks accurately
    M = 4096 
    k = np.arange(M)
    # Precompute the DFT matrix rows? No, just use fft.
    # But for speed in local search, precomputing the contribution of each bit is faster.
    # D[:, n] = exp(-2j * pi * k * n / M)
    # Let's precompute D as a matrix (M x 24)
    n_idx = np.arange(N)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def get_supnorm(coeffs):
        # coeffs: array of shape (24,) with values +/- 1
        # F = D @ coeffs
        F = D @ coeffs
        return np.max(np.abs(F))

    # Generate Golay-Rudin-Shapiro sequence
    def get_gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=float)

    gr_seq = get_gr_sequence(N)
    
    # Initial candidates
    candidates = [
        gr_seq,
        -gr_seq,
        gr_seq[::-1].copy(),
        -gr_seq[::-1].copy(),
        np.random.choice([-1, 1], N).astype(float)
    ]
    
    best_c = None
    best_val = float('inf')
    
    # Simulated Annealing / Local Search
    # We will run this for each candidate
    for init_c in candidates:
        current_c = init_c.copy()
        current_val = get_supnorm(current_c)
        
        if current_val < best_val:
            best_val = current_val
            best_c = current_c.copy()
            
        # Local search loop
        T = 1.0  # Temperature
        cooling = 0.999
        min_T = 1e-4
        
        while time.time() < deadline and T > min_T:
            # Pick a random move: flip 1 or 2 bits
            # Flipping 1 bit: 24 options
            # Flipping 2 bits: 24*23/2 = 276 options
            # Total 300 options. Very fast to evaluate.
            
            # We can batch evaluate all 1-flips
            # F_current = D @ current_c
            F_current = D @ current_c
            
            best_delta = 0
            best_move = None
            
            # Evaluate 1-flips
            for i in range(N):
                # F_new = F_current - 2 * current_c[i] * D[:, i]
                F_new = F_current - 2.0 * current_c[i] * D[:, i]
                new_val = np.max(np.abs(F_new))
                delta = new_val - current_val
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, -1) # Flip bit i
                    
            # If no improvement, try 2-flips? 
            # For speed, only try 2-flips if stuck or randomly.
            if best_delta >= 0:
                # Try a few random 2-flips
                for _ in range(10):
                    i = random.randint(0, N-1)
                    j = random.randint(0, N-1)
                    if i == j: continue
                    F_new = F_current - 2.0 * current_c[i] * D[:, i] - 2.0 * current_c[j] * D[:, j]
                    new_val = np.max(np.abs(F_new))
                    delta = new_val - current_val
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
            
            if best_move is None:
                T *= cooling
                continue
                
            # Accept move based on Metropolis criterion
            if best_delta < 0 or random.random() < np.exp(-best_delta / T):
                if isinstance(best_move[0], int) and len(best_move) == 2 and best_move[1] == -1:
                    # Single flip
                    current_c[best_move[0]] *= -1
                else:
                    # Double flip
                    current_c[best_move[0]] *= -1
                    current_c[best_move[1]] *= -1
                    
                current_val = current_val + best_delta
                
                if current_val < best_val:
                    best_val = current_val
                    best_c = current_c.copy()
                    
            T *= cooling
            
    return [int(x) for x in best_c]
