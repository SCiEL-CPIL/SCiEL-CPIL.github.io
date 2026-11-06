import random
import numpy as np
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Precompute all APs
    ap_indices_list = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices_list.append(indices)
    
    num_aps = len(ap_indices_list)
    
    # Build incidence matrix A (num_aps x N) where A[ap_idx, pos] = 1 if pos in ap
    A = np.zeros((num_aps, N), dtype=np.float64)
    for ap_idx, indices in enumerate(ap_indices_list):
        for idx in indices:
            A[ap_idx, idx - 1] = 1.0
    
    # Map position to APs containing it (for fast updates)
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices_list):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
    
    pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in pos_to_ap]
    
    def compute_initial_sums(s):
        # Use matrix-vector product for speed
        S = A @ s.astype(np.float64)
        return np.round(S).astype(np.int16)
    
    def build_hist(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        return hist, int(np.max(abs_S))
    
    def try_flip(j, s, S, hist, current_max, temp, rng):
        """Try flipping position j. Returns (accepted, new_max, delta_val)."""
        old_val = s[j]
        delta = -2 * old_val
        
        ap_idx_list = pos_to_ap_np[j]
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        for ap_idx in ap_idx_list:
            old_sum = S[ap_idx]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            S[ap_idx] = new_sum
            
            if new_abs > max_new_abs_in_affected:
                max_new_abs_in_affected = new_abs
        
        if hist[old_max] == 0:
            if max_new_abs_in_affected > old_max:
                new_max = max_new_abs_in_affected
            else:
                scan_result = old_max - 1
                while scan_result > 0 and hist[scan_result] == 0:
                    scan_result -= 1
                new_max = max(max_new_abs_in_affected, scan_result)
        else:
            new_max = max(old_max, max_new_abs_in_affected)
        
        cost_diff = new_max - old_max
        
        if cost_diff <= 0:
            accept = True
        else:
            if temp > 0.01:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
        
        if accept:
            s[j] = -old_val
            return True, new_max, delta
        else:
            for ap_idx in ap_idx_list:
                new_sum = S[ap_idx]
                old_sum = new_sum - delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                S[ap_idx] = old_sum
            
            return False, current_max, delta
    
    def local_search(s, S, hist, current_max, deadline, rng, max_steps=50000):
        """Simulated annealing with acceptance criteria."""
        temp = 4.0
        cooling_rate = 0.998
        steps = 0
        
        while time.time() < deadline and steps < max_steps:
            j = rng.randint(0, N - 1)
            
            accepted, new_max, _ = try_flip(j, s, S, hist, current_max, temp, rng)
            
            if accepted:
                current_max = new_max
            
            temp *= cooling_rate
            steps += 1
            
            if temp < 0.001:
                break
        
        return s, S, hist, current_max
    
    def greedy_improve(s, S, hist, current_max, deadline, rng, max_rounds=10):
        """Greedy hill-climbing: repeatedly try all positions, accept any improvement."""
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            for j in range(N):
                old_max = current_max
                old_val = s[j]
                delta = -2 * old_val
                
                ap_idx_list = pos_to_ap_np[j]
                max_new_abs_in_affected = 0
                
                for ap_idx in ap_idx_list:
                    old_sum = S[ap_idx]
                    new_sum = old_sum + delta
                    new_abs = abs(new_sum)
                    if new_abs > max_new_abs_in_affected:
                        max_new_abs_in_affected = new_abs
                
                if hist[old_max] == 0:
                    if max_new_abs_in_affected > old_max:
                        new_max = max_new_abs_in_affected
                    else:
                        scan_result = old_max - 1
                        while scan_result > 0 and hist[scan_result] == 0:
                            scan_result -= 1
                        new_max = max(max_new_abs_in_affected, scan_result)
                else:
                    new_max = max(old_max, max_new_abs_in_affected)
                
                if new_max < current_max:
                    # Accept: actually apply the flip
                    for ap_idx in ap_idx_list:
                        old_sum = S[ap_idx]
                        new_sum = old_sum + delta
                        
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                        
                        S[ap_idx] = new_sum
                    
                    s[j] = -old_val
                    current_max = new_max
                    improved = True
            
            if not improved:
                break
        
        return s, S, hist, current_max
    
    def perturb(s, S, hist, current_max, rng, num_flips=8):
        """Randomly flip some positions to escape local minima."""
        for _ in range(num_flips):
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            ap_idx_list = pos_to_ap_np[j]
            
            for ap_idx in ap_idx_list:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
            
            s[j] = -old_val
        
        # Recompute max
        abs_S = np.abs(S)
        current_max = int(np.max(abs_S))
        hist = np.zeros(N + 1, dtype=np.int32)
        np.add.at(hist, abs_S, 1)
        
        return s, S, hist, current_max
    
    def smooth_objective(s, beta=0.5):
        """Compute a smooth surrogate for the max absolute AP sum using log-sum-exp."""
        S = A @ s.astype(np.float64)
        abs_S = np.abs(S)
        # Log-sum-exp for numerical stability
        max_val = np.max(abs_S)
        if max_val == 0:
            return 0.0
        log_sum = max_val + np.log(np.sum(np.exp(abs_S - max_val)))
        return log_sum / beta
    
    # Multiple restarts with different strategies
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Initialize with random signs
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        S = compute_initial_sums(s)
        hist, current_max = build_hist(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # First: greedy improvement to get a good starting point
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + remaining_time * 0.15, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Then: simulated annealing
        remaining_time = deadline - time.time()
        if remaining_time > 3.0:
            sa_deadline = time.time() + remaining_time * 0.55
            s, S, hist, current_max = local_search(s, S, hist, current_max, sa_deadline, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Perturbation and re-optimization for escape from local minima
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            s, S, hist, current_max = perturb(s, S, hist, current_max, rng, num_flips=12)
            
            # Greedy + SA again
            if time.time() < deadline - 3.0:
                s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + (deadline - time.time()) * 0.25, rng)
            
            if time.time() < deadline - 2.0:
                s, S, hist, current_max = local_search(s, S, hist, current_max, time.time() + (deadline - time.time()) * 0.65, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Try a few more restarts if time permits
        if time.time() > deadline - 3.0:
            break
    
    # Final polish on the best sequence found
    if best_seq is not None and time.time() < deadline - 2.0:
        s_final = best_seq.copy()
        S_final = compute_initial_sums(s_final)
        hist_final, current_max_final = build_hist(S_final)
        
        # Greedy polish
        s_final, S_final, hist_final, current_max_final = greedy_improve(
            s_final, S_final, hist_final, current_max_final, deadline - 1.0, rng, max_rounds=15
        )
        
        # Short SA polish
        if time.time() < deadline - 1.0:
            s_final, S_final, hist_final, current_max_final = local_search(
                s_final, S_final, hist_final, current_max_final, deadline - 0.5, rng, max_steps=10000
            )
        
        if current_max_final < best_score:
            best_score = current_max_final
            best_seq = s_final.copy()
    
    return [int(x) for x in best_seq]
