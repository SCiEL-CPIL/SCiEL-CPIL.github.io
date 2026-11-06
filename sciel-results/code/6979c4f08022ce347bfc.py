import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Generate all APs of length >= 2
    all_aps = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                all_aps.append(terms)
    
    num_aps = len(all_aps)
    
    # Build incidence matrix M (num_aps x N) as float32
    M = np.zeros((num_aps, N), dtype=np.float32)
    for j, terms in enumerate(all_aps):
        for i in terms:
            M[j, i] = 1.0
    
    # Column norms (number of APs each position participates in)
    col_sums = M.sum(axis=0)  # length N
    
    def compute_sums(signs_f32):
        return M @ signs_f32
    
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))
    
    # Greedy set-cover: select critical APs that dominate the max
    # Strategy: start with random signs, find APs with large |sum|,
    # iteratively add APs that are "hard to satisfy"
    
    def select_critical_aps(signs_f32, target_count=2000):
        sums = compute_sums(signs_f32)
        abs_sums = np.abs(sums)
        
        # Start with APs that have the largest sums
        # Use a greedy approach: select APs that are most "constrained"
        selected = set()
        
        # First pass: select top APs by |sum|
        top_indices = np.argsort(-abs_sums)[:min(500, num_aps)]
        for idx in top_indices:
            selected.add(int(idx))
        
        # Second pass: iteratively add APs that are poorly covered
        # An AP is "poorly covered" if flipping a sign in it would reduce its |sum|
        # but that sign is in many other selected APs (so it's a bottleneck)
        
        # Simpler: select APs with high |sum| AND high "conflict"
        # Conflict = number of positions in this AP that are in many selected APs
        selected_arr = np.zeros(num_aps, dtype=bool)
        for idx in selected:
            selected_arr[idx] = True
        
        # Count how many selected APs each position is in
        pos_count = M[selected_arr] @ np.ones(N, dtype=np.float32)
        
        # Score each unselected AP: |sum| * avg(pos_count in AP)
        unselected_mask = ~selected_arr
        if unselected_mask.any():
            # For each unselected AP, compute average pos_count of its positions
            # pos_count[j] for AP j = M[j] @ pos_count
            ap_conflict = M @ pos_count
            score = abs_sums * ap_conflict
            score[selected_arr] = -1  # Exclude already selected
            
            # Add top unselected APs
            remaining = target_count - len(selected)
            if remaining > 0:
                unselected_indices = np.where(unselected_mask)[0]
                unselected_scores = score[unselected_indices]
                top_add = np.argsort(-unselected_scores)[:remaining]
                for idx in top_add:
                    selected.add(int(unselected_indices[idx]))
        
        # Ensure we have at least target_count
        if len(selected) < target_count:
            all_indices = set(range(num_aps)) - selected
            for idx in list(all_indices)[:target_count - len(selected)]:
                selected.add(idx)
        
        return sorted(selected)
    
    # Multi-restart with critical AP selection
    best_signs = None
    best_hard_max = N
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
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
        
        # Select critical APs based on current solution
        critical_idx = select_critical_aps(signs_f32, target_count=1500)
        critical_set = set(critical_idx)
        critical_arr = np.array(critical_idx, dtype=np.int32)
        
        # Build reduced matrix
        M_red = M[critical_arr]  # (len(critical) x N)
        
        # Local search on reduced constraint set
        def calc_soft_obj_red(sums_red, beta):
            abs_sums = np.abs(sums_red)
            max_val = np.max(abs_sums) / beta
            if max_val == 0:
                return 0.0
            shifted = abs_sums / beta - max_val
            sum_exp = np.sum(np.exp(shifted))
            return beta * (max_val + np.log(sum_exp))
        
        # SA Parameters
        beta = 1.0
        T = 5.0
        
        sums_red = M_red @ signs_f32
        current_obj = calc_soft_obj_red(sums_red, beta)
        
        # SA Loop on reduced set
        steps = 0
        while time.time() < deadline - 3:
            steps += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            
            # Update reduced sums
            col_i_red = M_red[:, i]
            delta_s_red = -2.0 * old_sign * col_i_red
            new_sums_red = sums_red + delta_s_red
            
            new_obj = calc_soft_obj_red(new_sums_red, beta)
            delta_obj = new_obj - current_obj
            
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
                sums_red = new_sums_red
                current_obj = new_obj
                
                # Periodic full verification
                if steps % 100 == 0:
                    full_sums = compute_sums(signs_f32)
                    current_hard_max = get_hard_max(full_sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        
                        # Update critical APs occasionally
                        if current_hard_max <= 4:
                            critical_idx = select_critical_aps(signs_f32, target_count=1500)
                            critical_arr = np.array(critical_idx, dtype=np.int32)
                            M_red = M[critical_arr]
                            sums_red = M_red @ signs_f32
                            current_obj = calc_soft_obj_red(sums_red, beta)
                        
            # Annealing
            if steps % 500 == 0:
                T *= 0.97
                beta *= 0.99
                if beta < 0.1:
                    beta = 0.1
                if T < 0.1:
                    T = 0.1
            
            # Early stop
            if best_hard_max <= 1:
                break
    
    # Final intensive local search on best solution
    if best_signs is not None and time.time() < deadline - 10:
        signs = best_signs.copy()
        signs_f32 = signs.astype(np.float32)
        
        # Select critical APs for final solution
        critical_idx = select_critical_aps(signs_f32, target_count=2000)
        critical_arr = np.array(critical_idx, dtype=np.int32)
        M_red = M[critical_arr]
        
        # Greedy local search: try flipping each position, keep if it reduces max
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            sums_red = M_red @ signs_f32
            current_max = int(np.max(np.abs(sums_red)))
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                old_sign = signs[i]
                signs_f32[i] = -signs_f32[i]
                new_sums_red = M_red @ signs_f32
                new_max = int(np.max(np.abs(new_sums_red)))
                
                if new_max < current_max:
                    signs[i] = -old_sign
                    current_max = new_max
                    improved = True
                else:
                    signs_f32[i] = old_sign
                    signs[i] = old_sign
        
        # Final full verification
        full_sums = compute_sums(signs_f32)
        final_max = get_hard_max(full_sums)
        if final_max < best_hard_max:
            best_hard_max = final_max
            best_signs = signs.copy()
    
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
