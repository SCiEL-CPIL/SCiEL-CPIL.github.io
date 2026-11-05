"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import random

def construct():
    n = 128
    target_low = 128
    target_high = 256
    
    # A is a subset of [0, 128]. Vector length 129.
    # x[i] = 1 if i in A.
    
    def objective(x):
        # x is a numpy array of length 129
        # Compute convolution to get ordered pair counts
        # C_s = sum_{a+b=s} x[a]x[b]
        # N_s (unordered, a<=b) = (C_s + D_s) / 2
        # D_s = 1 if s even and x[s/2] > 0, else 0 (for binary/continuous logic)
        
        # Convolve x with x
        # Result length is 257. Index s corresponds to sum s.
        C = np.convolve(x, x, mode='full')
        
        # Extract counts for s in [128, 256]
        # C[s] is the ordered count.
        # We need D_s for s in [128, 256]
        # D_s is non-zero only if s is even and s/2 is in [0, 128].
        # s/2 in [64, 128] for s in [128, 256].
        
        # Initialize N_s vector
        # We only care about s in [128, 256]
        counts = np.zeros(129) # indices 128..256 mapped to 0..128? No, let's just work on the slice.
        
        # Slice C for sums 128 to 256
        C_slice = C[128:257] # length 129
        
        # Calculate D_slice
        # s ranges from 128 to 256
        s_vals = np.arange(128, 257)
        # D_s is x[s/2] if s is even, else 0
        # s/2 must be integer.
        # Create array for s/2
        half_s = s_vals / 2.0
        # Check if integer
        is_even = (s_vals % 2 == 0)
        
        # For even s, index is s/2.
        # We need to gather x[s/2] for even s.
        # x has length 129 (indices 0..128).
        # s/2 ranges from 64 to 128.
        
        D_slice = np.zeros(129)
        even_indices = np.where(is_even)[0]
        for idx in even_indices:
            s = s_vals[idx]
            a = s // 2
            # a is in [64, 128], which is valid index for x
            D_slice[idx] = x[a]
            
        # N_s = (C_s + D_s) / 2
        N_slice = (C_slice + D_slice) / 2.0
        
        # We want to minimize max(N_slice) subject to N_slice >= 1 (or > 0.5 for continuous)
        # For DE, we treat x as continuous [0,1].
        # If N_s < 0.5, it's effectively uncovered.
        
        max_count = np.max(N_slice)
        
        # Penalty for uncovered sums
        # If any N_s < 0.5, add large penalty
        uncovered_count = np.sum(N_slice < 0.5)
        penalty = 1000 * uncovered_count
        
        return max_count + penalty

    # Initial bounds
    bounds = [(0, 1)] * 129
    
    # Run Differential Evolution
    # Time limit 60s. Let's set maxiter to something reasonable and check time.
    # popsize=15, maxiter=200 -> 3000 evaluations.
    # Each eval is ~100us (convolve is fast). 3000 * 100us = 0.3s.
    # We can afford more iterations.
    
    start_time = time.time()
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=20, 
            tol=0.01,
            seed=42,
            polish=False # We will polish manually
        )
    except Exception:
        # Fallback to simple heuristic if DE fails or times out
        res = None

    # Convert to binary
    if res and res.success:
        x_opt = res.x
    else:
        # Fallback: Start with a known good set, e.g., [64, 128]
        x_opt = np.zeros(129)
        x_opt[64:129] = 1.0
        
    # Threshold to binary
    A_binary = (x_opt > 0.5).astype(int)
    A_set = set(np.where(A_binary == 1)[0].tolist())
    
    # Local Search Refinement
    # 1. Ensure coverage
    # 2. Minimize max count
    
    def get_stats(A_set):
        max_count = 0
        uncovered = 0
        for s in range(target_low, target_high + 1):
            count = 0
            a_min = max(0, s - 128)
            a_max = min(128, s // 2)
            for a in range(a_min, a_max + 1):
                if a in A_set and (s - a) in A_set:
                    count += 1
            if count == 0:
                uncovered += 1
            else:
                if count > max_count:
                    max_count = count
        return max_count, uncovered

    current_max, current_uncovered = get_stats(A_set)
    
    # If uncovered, try to add elements to fix
    # If covered, try to remove/swap to reduce max
    
    time_limit = 55
    best_A = set(A_set)
    best_max = current_max
    
    # Simple local search
    for _ in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        if current_uncovered > 0:
            # Try to add a random element
            candidates = [i for i in range(129) if i not in A_set]
            if not candidates:
                break
            add_elem = random.choice(candidates)
            new_A = A_set | {add_elem}
            new_max, new_uncovered = get_stats(new_A)
            if new_uncovered < current_uncovered:
                A_set = new_A
                current_max, current_uncovered = new_max, new_uncovered
            elif new_uncovered == current_uncovered and new_max < current_max:
                A_set = new_A
                current_max, current_uncovered = new_max, new_uncovered
        else:
            # Try to remove an element
            if len(A_set) > 1:
                remove_elem = random.choice(list(A_set))
                new_A = A_set - {remove_elem}
                new_max, new_uncovered = get_stats(new_A)
                if new_uncovered == 0 and new_max <= current_max:
                    A_set = new_A
                    current_max = new_max
            # Try to swap
            if len(A_set) > 0:
                remove_elem = random.choice(list(A_set))
                candidates = [i for i in range(129) if i not in A_set]
                if candidates:
                    add_elem = random.choice(candidates)
                    new_A = (A_set - {remove_elem}) | {add_elem}
                    new_max, new_uncovered = get_stats(new_A)
                    if new_uncovered == 0 and new_max < current_max:
                        A_set = new_A
                        current_max = new_max
                        
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)
            
    return list(best_A)
# EVOLVE-BLOCK-END
