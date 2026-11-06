import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    N = 256

    # 1. Precompute ALL APs and inverse mapping
    ap_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    ap_lengths = np.array([len(lst) for lst in ap_indices], dtype=np.int32)
    
    # Build incidence matrix for gradient computation
    # incidence[ap_idx, position] = 1 if position is in AP, else 0
    incidence = np.zeros((num_aps, N), dtype=np.float64)
    for ap_idx, idx_list in enumerate(ap_indices):
        for idx in idx_list:
            incidence[ap_idx, idx] = 1.0

    # 2. Initialization Functions
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    def perturbed_init(base_s, rng, num_flips):
        s = base_s.copy()
        for _ in range(num_flips):
            i = rng.randint(0, N - 1)
            s[i] = -s[i]
        return s

    # 3. Core Optimization with histogram tracking
    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        max_possible_sum = N
        hist = np.zeros(max_possible_sum + 2, dtype=np.int32)
        for val in np.abs(sums):
            hist[val] += 1
        
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
        
        indices_list = list(range(N))
        target_D = max(0, current_max - 1)
        T = 1.0
        T_min = 0.01
        cooling = 0.95
        stuck_counter = 0
        max_stuck = 500
        perturbation_cooldown = 0
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                if perturbation_cooldown > 0:
                    perturbation_cooldown -= 1
                    continue
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                new_sums_affected = sums[affected] + delta
                new_abs_affected = np.abs(new_sums_affected)
                
                violates_target = np.any(new_abs_affected > target_D)
                violates_current_max = np.any(new_abs_affected > current_max)
                
                if violates_current_max:
                    continue
                
                if not violates_target:
                    old_abs_affected = np.abs(sums[affected])
                    
                    for j, ap_idx in enumerate(affected):
                        old_abs = old_abs_affected[j]
                        new_abs = new_abs_affected[j]
                        if old_abs != new_abs:
                            hist[old_abs] -= 1
                            hist[new_abs] += 1
                    
                    sums[affected] = new_sums_affected
                    s[i] = new_val
                    stuck_counter = 0
                    
                    new_global_max = get_max_from_hist()
                    if new_global_max < current_max:
                        current_max = new_global_max
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                        target_D = max(0, current_max - 1)
                        T = 1.0
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                        
                else:
                    max_violation = int(np.max(new_abs_affected - target_D))
                    p_accept = 0.05 + 0.15 * (stuck_counter / max_stuck) - 0.01 * max_violation
                    p_accept = max(0.0, min(0.5, p_accept))
                    if rng.random() < p_accept:
                        old_abs_affected = np.abs(sums[affected])
                        
                        for j, ap_idx in enumerate(affected):
                            old_abs = old_abs_affected[j]
                            new_abs = new_abs_affected[j]
                            if old_abs != new_abs:
                                hist[old_abs] -= 1
                                hist[new_abs] += 1
                        
                        sums[affected] = new_sums_affected
                        s[i] = new_val
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                
                if stuck_counter >= max_stuck:
                    num_perturb = rng.randint(2, 5)
                    perturbation_indices = rng.sample(range(N), num_perturb)
                    
                    s_backup = s.copy()
                    sums_backup = sums.copy()
                    hist_backup = hist.copy()
                    
                    for pi in perturbation_indices:
                        old_val = s[pi]
                        delta = -2 * old_val
                        new_val = -old_val
                        affected = aps_containing_np[pi]
                        new_sums_affected = sums[affected] + delta
                        new_abs_affected = np.abs(new_sums_affected)
                        
                        if not np.any(new_abs_affected > current_max):
                            old_abs_affected = np.abs(sums[affected])
                            for j, ap_idx in enumerate(affected):
                                old_abs = old_abs_affected[j]
                                new_abs = new_abs_affected[j]
                                if old_abs != new_abs:
                                    hist[old_abs] -= 1
                                    hist[new_abs] += 1
                            sums[affected] = new_sums_affected
                            s[pi] = new_val
                    
                    new_global_max = get_max_from_hist()
                    if new_global_max < current_max:
                        current_max = new_global_max
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                        target_D = max(0, current_max - 1)
                        stuck_counter = 0
                        T = 1.0
                    else:
                        s = s_backup
                        sums = sums_backup
                        hist = hist_backup
                        stuck_counter = 0
                        T *= cooling
                        if T < T_min:
                            T = T_min
                    perturbation_cooldown = 10

        return best_s, best_score

    # 4. Gradient-based continuous optimization
    def sgd_optimize(time_budget, rng):
        """Use SGD on continuous variables to find good basins"""
        # Initialize with random continuous values in [-1, 1]
        x = np.random.uniform(-1, 1, size=N).astype(np.float64)
        
        lr = 0.01
        lr_decay = 0.999
        momentum = 0.9
        v = np.zeros(N, dtype=np.float64)
        
        best_x = x.copy()
        best_discreet_score = N
        
        while time.time() - start_time < time_budget:
            # Compute AP sums: sums[ap] = sum of x[idx] for idx in ap
            # Using incidence matrix: sums = incidence @ x
            sums = incidence @ x
            
            # Loss: sum of squares of AP sums (smooth surrogate)
            # Gradient: 2 * incidence.T @ sums
            grad = 2.0 * (incidence.T @ sums)
            
            # Momentum update
            v = momentum * v - lr * grad
            x = x + v
            
            # Keep x bounded (soft clipping via tanh-like behavior)
            # Actually, let's just clip to [-1, 1] for now
            x = np.clip(x, -1.0, 1.0)
            
            # Discretize and evaluate
            x_disc = np.where(x >= 0, 1, -1).astype(np.int8)
            
            # Compute max |sum| for discretized version
            sums_disc = np.zeros(num_aps, dtype=np.int16)
            for r, idx_list in enumerate(ap_indices_np):
                sums_disc[r] = np.sum(x_disc[idx_list])
            disc_max = int(np.max(np.abs(sums_disc)))
            
            if disc_max < best_discreet_score:
                best_discreet_score = disc_max
                best_x = x.copy()
            
            # Decay learning rate
            lr *= lr_decay
            if lr < 0.001:
                break
        
        # Discretize best_x
        best_disc = np.where(best_x >= 0, 1, -1).astype(np.int8)
        return best_disc, best_discreet_score

    # 5. Main Loop with Restarts
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run 1: Thue-Morse (25%)
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_left * 0.25, rng)
    if score1 < best_score_overall:
        best_score_overall = score1
        best_s_overall = s1
        
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 2: Gradient-based SGD (25%)
        s2, score2 = sgd_optimize(time_left * 0.25, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2.copy()
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 3: Random (20%)
        rand1 = random_init(N, rng)
        s3, score3 = optimize(rand1, time_left * 0.20, rng)
        if score3 < best_score_overall:
            best_score_overall = score3
            best_s_overall = s3
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 4: Perturbed Thue-Morse (15%)
        pert_tm = perturbed_init(tm, rng, 5)
        s4, score4 = optimize(pert_tm, time_left * 0.15, rng)
        if score4 < best_score_overall:
            best_score_overall = score4
            best_s_overall = s4
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 5: Another SGD run with different init (10%)
        s5, score5 = sgd_optimize(time_left * 0.10, rng)
        if score5 < best_score_overall:
            best_score_overall = score5
            best_s_overall = s5.copy()
            
    # Final Polish
    time_left = time_limit - (time.time() - start_time)
    if time_left > 0.5 and best_s_overall is not None:
        s_final, score_final = optimize(best_s_overall, time_left * 0.9, rng)
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final

    return [int(x) for x in best_s_overall]
