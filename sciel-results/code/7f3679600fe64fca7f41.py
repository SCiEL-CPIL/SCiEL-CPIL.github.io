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
    M = np.zeros((num_aps, N), dtype=np.float32)
    # Also store the end index for each AP for greedy init
    ap_end_indices = []
    for j, terms in enumerate(ap_indices):
        for i in terms:
            M[j, i] = 1.0
        ap_end_indices.append(terms[-1])
        
    # Helper to compute sums
    def compute_sums(signs):
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
    
    # Precompute APs ending at each index for Greedy Initialization
    # ap_end_indices[k] is the last index of AP k
    # We need a list of AP indices for each end position
    aps_ending_at = [[] for _ in range(N)]
    for k, end_idx in enumerate(ap_end_indices):
        aps_ending_at[end_idx].append(k)

    def greedy_init(rng):
        signs = np.zeros(N, dtype=np.int8)
        # We fill signs[0..N-1]
        # For each i, choose s in {1, -1} to minimize max |sum| of APs ending at i
        # Note: APs ending at i use signs[0..i]
        # We can maintain a running sum array for all APs?
        # sums[j] = sum of signs in AP j so far?
        # If we update signs[i], we update sums for all APs containing i.
        # For greedy, we only care about APs ending at i.
        # But their sums depend on previous signs.
        
        # Let's just compute the sums for APs ending at i on the fly.
        # It's O(num_aps_ending_at_i * i).
        
        for i in range(N):
            best_s = 1
            best_cost = float('inf')
            second_best_s = -1
            second_best_cost = float('inf')
            
            for s in [1, -1]:
                signs[i] = s
                # Calculate max abs sum for APs ending at i
                # These APs are fully defined by signs[0..i]
                max_sum = 0
                for ap_idx in aps_ending_at[i]:
                    # Sum of M[ap_idx, 0..i] * signs[0..i]
                    # Since M[ap_idx, i+1..N-1] is 0, we can just dot M[ap_idx] with signs
                    # But signs[i+1..] are 0.
                    # So np.dot(M[ap_idx], signs) works.
                    # However, M is float32, signs is int8.
                    # We need to be careful with types.
                    s_val = np.dot(M[ap_idx], signs.astype(np.float32))
                    abs_val = abs(s_val)
                    if abs_val > max_sum:
                        max_sum = abs_val
                
                if max_sum < best_cost:
                    second_best_s = best_s
                    second_best_cost = best_cost
                    best_s = s
                    best_cost = max_sum
                elif max_sum < second_best_cost:
                    second_best_s = s
                    second_best_cost = max_sum
            
            # Tie-breaking with randomness
            # If costs are close, pick randomly
            if best_cost == second_best_cost:
                if rng.random() < 0.5:
                    signs[i] = second_best_s
                else:
                    signs[i] = best_s
            else:
                signs[i] = best_s
                
        return signs

    def calc_soft_obj(sums, beta):
        abs_sums = np.abs(sums)
        max_val = np.max(abs_sums) / beta
        if max_val == 0:
            return 0.0
        shifted = abs_sums / beta - max_val
        sum_exp = np.sum(np.exp(shifted))
        return beta * (max_val + np.log(sum_exp))

    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize with Greedy
        signs = greedy_init(rng)
        signs_f32 = signs.astype(np.float32)
        sums = compute_sums(signs_f32)
        
        current_hard_max = get_hard_max(sums)
        
        if current_hard_max < best_hard_max:
            best_hard_max = current_hard_max
            best_signs = signs.copy()

        # SA Parameters
        # Start with lower T since we start from a good point
        beta = 0.5 
        T = 1.0
        
        current_obj = calc_soft_obj(sums, beta)
        
        # SA Loop
        steps = 0
        max_steps = 200000 # High number, limited by time
        
        while time.time() < deadline - 2:
            steps += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            
            # Update sums
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
                if delta_obj / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_obj / T)
            
            if accept:
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
            if steps % 500 == 0:
                T *= 0.90
                beta *= 0.95
                if beta < 0.1:
                    beta = 0.1
                if T < 0.05:
                    T = 0.05
                    
            if best_hard_max <= 1:
                break
                
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
