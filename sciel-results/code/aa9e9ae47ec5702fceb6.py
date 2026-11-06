import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # 1. Precompute APs and construct the dense incidence matrix
    # M[j, i] = 1 if position i is in AP j, else 0
    # We include APs of length >= 2.
    # Note: Length 1 APs have sum +/- 1, so max is at least 1.
    
    ap_indices = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Build dense matrix M (num_aps x N)
    # Using int8 or bool to save memory, but float64 is needed for matmul speed?
    # Actually, M is binary. We can store as float32 or float64.
    # N=256, num_aps ~ sum_{d=1}^{255} (256-d) = 255*256 - 255*256/2 = 32640.
    # 32640 * 256 * 8 bytes ~ 67 MB. This is fine.
    
    M = np.zeros((num_aps, N), dtype=np.float32)
    for j, terms in enumerate(ap_indices):
        for i in terms:
            M[j, i] = 1.0
            
    # Helper to compute sums
    def compute_sums(signs):
        # signs is int8 or int16 array
        # M is float32
        # Result is float32
        return M @ signs.astype(np.float32)
        
    # Helper to get hard max
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))

    best_signs = None
    best_hard_max = N # Worst case

    # Multi-restart SA
    # We will use a soft objective for the SA dynamics: F = beta * log(sum(exp(|S|/beta)))
    # And track the hard max for the final result.
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random signs
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        signs_f32 = signs.astype(np.float32)
        sums = compute_sums(signs_f32)
        
        current_hard_max = get_hard_max(sums)
        
        # SA Parameters
        # Beta controls the sharpness of the log-sum-exp approximation to the max.
        # As beta -> 0, F -> max(|S|).
        # T controls the acceptance of worse moves.
        beta = 1.0
        T = 5.0
        
        # Initial soft objective
        def calc_soft_obj(sums, beta):
            # Stable log sum exp of |sums|/beta
            abs_sums = np.abs(sums)
            max_val = np.max(abs_sums) / beta
            if max_val == 0:
                return 0.0
            shifted = abs_sums / beta - max_val
            sum_exp = np.sum(np.exp(shifted))
            return beta * (max_val + np.log(sum_exp))
            
        current_obj = calc_soft_obj(sums, beta)
        
        # SA Loop
        steps = 0
        max_steps = 200000 # High number, limited by time
        
        while time.time() < deadline - 2:
            steps += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            
            # Update sums: S_new = S_old - 2 * old_sign * M[:, i]
            # M[:, i] is the column for position i
            col_i = M[:, i]
            
            # Calculate new sums
            # We can do this in-place or compute delta
            # Delta S = -2 * old_sign * col_i
            delta_s = -2.0 * old_sign * col_i
            
            # New sums = sums + delta_s
            new_sums = sums + delta_s
            
            # Calculate new soft objective
            new_obj = calc_soft_obj(new_sums, beta)
            
            delta_obj = new_obj - current_obj
            
            # Metropolis Criterion
            accept = False
            if delta_obj < 0:
                accept = True
            else:
                # Avoid overflow
                if delta_obj / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_obj / T)
            
            if accept:
                # Apply move
                signs[i] = -old_sign
                signs_f32[i] = -signs_f32[i]
                sums = new_sums
                current_obj = new_obj
                
                # Check hard max periodically
                if steps % 50 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        
            # Annealing
            # Slowly decrease T and beta
            if steps % 1000 == 0:
                T *= 0.95
                beta *= 0.98
                if beta < 0.1:
                    beta = 0.1
                if T < 0.1:
                    T = 0.1
                    
            # Early stop if we hit 1 (theoretical minimum for max AP sum > 1 is 1, 
            # and total score is max(1, max_ap_sum). If max_ap_sum is 1, score is 1.
            # If max_ap_sum is 0, score is 1. So 1 is the lower bound.)
            if best_hard_max <= 1:
                break
                
    # If no best_signs found, return random
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
