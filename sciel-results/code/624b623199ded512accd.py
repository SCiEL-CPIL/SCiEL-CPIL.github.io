"""Combined program: incremental SA with histogram tracking (inspiration 1) + multi-restart + polish."""
import time
import random
import numpy as np
import math

N = 256

def _build_aps():
    """Build all arithmetic progressions of length >= 1 contained in [0, N-1]."""
    ap_indices = []
    pos_to_ap = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for a in range(N):
            # Positions: a, a+d, a+2d, ... < N
            positions = list(range(a, N, d))
            if len(positions) < 1:
                continue
            # All sub-APs starting at positions[i] with step d
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    ap = positions[i:j+1]
                    ap_idx = len(ap_indices)
                    ap_indices.append(ap)
                    for pos in ap:
                        pos_to_ap[pos].append(ap_idx)
    
    return ap_indices, pos_to_ap

# Precompute at module load time
_ap_indices, _pos_to_ap = _build_aps()
_num_aps = len(_ap_indices)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Try a few deterministic initializations first
    inits = []
    # Alternating
    inits.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8))
    # Random seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        rng2 = random.Random(seed)
        inits.append(np.array([1 if rng2.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8))
    
    # Evaluate all initializations
    init_scores = []
    for s in inits:
        S = np.zeros(_num_aps, dtype=np.int16)
        for ap_idx, ap in enumerate(_ap_indices):
            S[ap_idx] = np.sum(s[ap])
        abs_S = np.abs(S)
        current_max = int(np.max(abs_S))
        init_scores.append((current_max, s, S, abs_S))
    
    # Sort by score
    init_scores.sort(key=lambda x: x[0])
    
    for current_max, s, S, abs_S in init_scores:
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
    
    # Build histogram for the best initial solution
    def build_hist(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        return hist, abs_S
    
    # Main search: iterate over initializations, run SA on each
    for init_idx, (init_max, s_init, S_init, _) in enumerate(init_scores):
        if time.time() >= deadline - 5.0:
            break
        
        s = s_init.copy()
        S = S_init.copy()
        hist, abs_S = build_hist(S)
        current_max = init_max
        
        # Simulated Annealing
        temp = 3.0
        cooling_rate = 0.999
        steps_in_restart = 0
        max_steps = 8000
        
        while time.time() < deadline - 5.0 and steps_in_restart < max_steps:
            if temp < 0.005:
                break
            
            j = rng.randint(0, N - 1)
            
            old_val = s[j]
            delta = -2 * old_val
            
            ap_indices_for_j = _pos_to_ap[j]
            
            old_max = current_max
            max_new_abs_in_affected = 0
            
            for ap_idx in ap_indices_for_j:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
                
                if new_abs > max_new_abs_in_affected:
                    max_new_abs_in_affected = new_abs
            
            # Determine new max
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
                if temp > 0.001:
                    p = math.exp(-cost_diff / temp)
                    accept = (rng.random() < p)
                else:
                    accept = False
            
            if accept:
                s[j] = -old_val
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_seq = s.copy()
            else:
                # Revert
                for ap_idx in ap_indices_for_j:
                    new_sum = S[ap_idx]
                    old_sum = new_sum - delta
                    
                    old_abs = abs(old_sum)
                    new_abs = abs(new_sum)
                    
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
                    
                    S[ap_idx] = old_sum
            
            temp *= cooling_rate
            steps_in_restart += 1
        
        # Final polish pass
        if time.time() < deadline - 3.0:
            polish_deadline = deadline - 3.0
            polish_temp = 0.3
            polish_cooling = 0.995
            polish_steps = 0
            max_polish = 3000
            
            while time.time() < polish_deadline and polish_steps < max_polish:
                if polish_temp < 0.001:
                    break
                
                j = rng.randint(0, N - 1)
                
                old_val = s[j]
                delta = -2 * old_val
                
                ap_indices_for_j = _pos_to_ap[j]
                
                old_max = current_max
                max_new_abs_in_affected = 0
                
                for ap_idx in ap_indices_for_j:
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
                    p = math.exp(-cost_diff / polish_temp)
                    accept = (rng.random() < p)
                
                if accept:
                    s[j] = -old_val
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_seq = s.copy()
                else:
                    for ap_idx in ap_indices_for_j:
                        new_sum = S[ap_idx]
                        old_sum = new_sum - delta
                        
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        
                        hist[new_abs] -= 1
                        hist[old_abs] += 1
                        
                        S[ap_idx] = old_sum
                
                polish_temp *= polish_cooling
                polish_steps += 1
    
    # Final local search on best solution
    if best_seq is not None and time.time() < deadline - 1.0:
        s = best_seq.copy()
        S = np.zeros(_num_aps, dtype=np.int16)
        for ap_idx, ap in enumerate(_ap_indices):
            S[ap_idx] = np.sum(s[ap])
        hist, abs_S = build_hist(S)
        current_max = int(np.max(abs_S))
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for j in range(N):
                old_val = s[j]
                delta = -2 * old_val
                
                ap_indices_for_j = _pos_to_ap[j]
                
                old_max = current_max
                max_new_abs_in_affected = 0
                
                for ap_idx in ap_indices_for_j:
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
                
                if new_max < current_max:
                    s[j] = -old_val
                    current_max = new_max
                    improved = True
                else:
                    # Revert
                    for ap_idx in ap_indices_for_j:
                        new_sum = S[ap_idx]
                        old_sum = new_sum - delta
                        
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        
                        hist[new_abs] -= 1
                        hist[old_abs] += 1
                        
                        S[ap_idx] = old_sum
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
    
    return [int(x) for x in best_seq]
