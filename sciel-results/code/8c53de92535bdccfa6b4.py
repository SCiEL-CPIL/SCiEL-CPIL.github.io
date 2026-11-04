"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Strategy:
    1. Binary search space (0/1) is sufficient for minimizing max overlap in this linear setup.
    2. Use Thue-Morse initialization for a good starting point with low correlations.
    3. Perform Minimax Hill Climbing with efficient O(N) delta updates for the correlation array.
    4. Use Simulated Annealing style acceptance for escaping local minima.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Initialize with Thue-Morse sequence
    # Thue-Morse: t_n = popcount(n) % 2
    # We need exactly N/2 ones.
    # Thue-Morse of length 100 might not have exactly 50 ones.
    # Let's generate it and fix the count.
    
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=float)

    h = thue_morse(N)
    
    # Adjust to have exactly N//2 ones
    num_ones = int(np.sum(h))
    target_ones = N // 2
    
    if num_ones > target_ones:
        # Remove (num_ones - target_ones) ones
        ones_indices = np.where(h == 1)[0]
        to_remove = np.random.choice(ones_indices, num_ones - target_ones, replace=False)
        h[to_remove] = 0
    elif num_ones < target_ones:
        # Add (target_ones - num_ones) ones
        zeros_indices = np.where(h == 0)[0]
        to_add = np.random.choice(zeros_indices, target_ones - num_ones, replace=False)
        h[to_add] = 1

    # Compute initial correlation array C
    # C[k] for k in -(N-1) ... N-1
    # We only care about k != 0.
    # np.correlate(h, 1-h, mode='full')
    # Index N-1 is k=0.
    
    def compute_corr(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        return c

    c = compute_corr(h)
    
    # Mask out k=0 (index N-1)
    mask = np.ones(len(c), dtype=bool)
    mask[N-1] = False
    
    best_c = c.copy()
    best_max = np.max(c[mask])
    best_h = h.copy()
    
    current_c = c.copy()
    current_max = best_max
    current_h = h.copy()
    
    # Precompute indices of ones and zeros for fast swap selection
    # We'll update these lists or just use np.where inside the loop (slower but simpler)
    # For N=100, np.where is fast enough.
    
    T = 5.0  # Initial temperature
    T_min = 0.05
    cooling = 0.9995
    
    ones = np.where(current_h == 1)[0]
    zeros = np.where(current_h == 0)[0]
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min # or break
        
        # Get current ones and zeros
        ones = np.where(current_h == 1)[0]
        zeros = np.where(current_h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        # Proposed swap: h[i] 1->0, h[j] 0->1
        # Calculate delta in C for this swap
        # Delta C_k = (change due to i) + (change due to j)
        
        # Change due to i (1->0):
        # Term 1 (i as first): h_i(1-h_{i+k}) -> 0. Delta: -h_i(1-h_{i+k}) = -(1-h_{i+k})
        # Term 2 (i as second): h_{i-k}(1-h_i) -> h_{i-k}(1-0)=h_{i-k}. Delta: h_{i-k} - h_{i-k}(1-h_i) = h_{i-k}*h_i = h_{i-k}
        # So Delta_i(k) = -(1-h_{i+k}) + h_{i-k} (with valid index checks)
        
        # Change due to j (0->1):
        # Term 1 (j as first): 0 -> (1-h_{j+k}). Delta: (1-h_{j+k})
        # Term 2 (j as second): h_{j-k}(1-0) -> h_{j-k}(1-1)=0. Delta: -h_{j-k}
        # So Delta_j(k) = (1-h_{j+k}) - h_{j-k}
        
        # Total Delta(k) = Delta_i(k) + Delta_j(k)
        
        # We need to apply this to current_c.
        # Instead of vectorizing the index checks which is tricky in numpy,
        # we can just compute the new C for the affected shifts?
        # No, it affects ALL shifts.
        
        # Let's use a helper to compute the delta vector efficiently.
        # Since N=100, a python loop over k is ~100 ops.
        # Let's try to vectorize it.
        
        # Valid k for i:
        # i+k in [0, N-1] => k in [-i, N-1-i]
        # i-k in [0, N-1] => k in [i-(N-1), i]
        # Intersection: k in [max(-i, i-N+1), min(N-1-i, i)]
        
        # This is complex to vectorize with variable bounds per i,j.
        # Given N=100, let's just do a small numpy operation for the update.
        
        # Actually, recomputing full correlate is O(N^2) = 10^4.
        # Python loop over 100 shifts is O(N) = 100.
        # Python loop overhead is high.
        
        # Alternative: Just use np.correlate for every step?
        # 120s / 10000 ops ~ 12000 steps? Too few.
        
        # Let's stick to the O(N) python loop for delta calculation.
        # It's 100 iterations of simple math.
        
        delta_c = np.zeros(len(current_c))
        
        # Helper to get h[idx] if valid else 0
        # We can pre-extract h_i, h_j
        hi = current_h[i]
        hj = current_h[j]
        
        # We are swapping i (1) and j (0)
        # So new_h[i]=0, new_h[j]=1
        
        # Iterate k
        # k range in correlate 'full' is -(N-1) to N-1.
        # Index in array: k + (N-1)
        
        for k in range(-(N-1), N):
            idx = k + (N-1)
            
            d = 0.0
            
            # Contribution from i
            # i+k valid?
            if 0 <= i+k < N:
                # Term: - (1 - h[i+k])
                d -= (1.0 - current_h[i+k])
            
            # i-k valid?
            if 0 <= i-k < N:
                # Term: + h[i-k]
                d += current_h[i-k]
                
            # Contribution from j
            # j+k valid?
            if 0 <= j+k < N:
                # Term: + (1 - h[j+k])
                d += (1.0 - current_h[j+k])
                
            # j-k valid?
            if 0 <= j-k < N:
                # Term: - h[j-k]
                d -= current_h[j-k]
            
            delta_c[idx] = d
        
        new_c = current_c + delta_c
        new_max = np.max(new_c[mask])
        
        # Acceptance
        if new_max < current_max or random.random() < np.exp(-(new_max - current_max) / T):
            # Accept
            current_h[i] = 0.0
            current_h[j] = 1.0
            current_c = new_c
            current_max = new_max
            
            if new_max < best_max:
                best_max = new_max
                best_h = current_h.copy()
                best_c = new_c.copy()
        else:
            # Reject
            pass
        
        T *= cooling

    # Final Polish: Greedy local search
    # Try all single bit flips/swaps to see if we can improve further
    improved = True
    while improved:
        improved = False
        ones = np.where(best_h == 1)[0]
        zeros = np.where(best_h == 0)[0]
        
        # To save time, we don't try all N^2 swaps.
        # We try swaps that reduce the current worst shift.
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        c_best = compute_corr(best_h)
        worst_idx = np.argmax(c_best[mask])
        worst_k = worst_idx - (N-1)
        
        # Find bad pairs for this k
        # We want h[i] != h[i+k]
        # Try to fix them
        
        # Simple approach: Try random swaps again but with T=0 (greedy)
        # Limit iterations
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
                
            i = ones[np.random.randint(len(ones))]
            j = zeros[np.random.randint(len(ones)) if len(ones)>0 else 0] # Fix: use zeros
            # Corrected:
            j = zeros[np.random.randint(len(zeros))]
            
            # Calculate delta
            delta_c = np.zeros(len(c_best))
            for k in range(-(N-1), N):
                idx = k + (N-1)
                d = 0.0
                if 0 <= i+k < N: d -= (1.0 - best_h[i+k])
                if 0 <= i-k < N: d += best_h[i-k]
                if 0 <= j+k < N: d += (1.0 - best_h[j+k])
                if 0 <= j-k < N: d -= best_h[j-k]
                delta_c[idx] = d
            
            new_c = c_best + delta_c
            new_max = np.max(new_c[mask])
            
            if new_max < best_max:
                best_h[i] = 0.0
                best_h[j] = 1.0
                best_max = new_max
                c_best = new_c
                improved = True
                # Update ones/zeros
                ones = np.where(best_h == 1)[0]
                zeros = np.where(best_h == 0)[0]
                if len(ones)==0 or len(zeros)==0: break
                break # Restart loop to re-evaluate worst_k? Or continue?
                # Let's break to re-identify worst shift

    return best_h

# EVOLVE-BLOCK-END
