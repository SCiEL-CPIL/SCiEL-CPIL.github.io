import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

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
    M = np.zeros((num_aps, N), dtype=np.float32)
    for j, terms in enumerate(ap_indices):
        for i in terms:
            M[j, i] = 1.0
            
    # Precompute column norms for potential gradient-based moves
    # col_sums[j] = M[j, i]
    
    # Helper to compute sums
    def compute_sums(signs_f32):
        return M @ signs_f32
        
    # Helper to get hard max
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))

    best_signs = None
    best_hard_max = N # Worst case

    # Multi-restart SA with better initialization and cooling
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    # We'll run multiple restarts. Each restart has a time budget.
    # If we find a hard max of 1, we stop.
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random signs
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        signs_f32 = signs.astype(np.float32)
        sums = compute_sums(signs_f32)
        
        current_hard_max = get_hard_max(sums)
        
        if current_hard_max < best_hard_max:
            best_hard_max = current_hard_max
            best_signs = signs.copy()
            
        # SA Parameters
        # We will use a simpler objective: minimize max(|S|) directly or approx.
        # Using soft max is good.
        # beta starts at 1.0 and decreases.
        # T starts at 1.0 and decreases.
        
        beta = 1.0
        T = 1.0
        
        # Initial soft objective
        def calc_soft_obj(sums, beta):
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
        # Allocate time per restart
        restart_time_budget = 5.0 
        restart_deadline = time.time() + restart_time_budget
        
        while time.time() < restart_deadline and time.time() < deadline - 2:
            steps += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            
            # Update sums: S_new = S_old - 2 * old_sign * M[:, i]
            col_i = M[:, i]
            delta_s = -2.0 * old_sign * col_i
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
                if steps % 20 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        
            # Annealing
            # Fast cooling
            if steps % 50 == 0:
                T *= 0.95
                beta *= 0.99
                if beta < 0.1:
                    beta = 0.1
                if T < 0.01:
                    T = 0.01
                    
            # Early stop if we hit 1
            if best_hard_max <= 1:
                break
                
        # If we broke out of the restart loop early due to finding 1, break outer loop
        if best_hard_max <= 1:
            break

    # If no best_signs found, return random
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
