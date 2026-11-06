import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    def get_precomputed(n):
        triples = list(combinations(range(n), 3))
        t_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        t_to_k4 = [[] for _ in triples]
        t_to_k5 = [[] for _ in triples]
        
        for i, k4 in enumerate(k4s):
            ts = [t_idx[t] for t in combinations(k4, 3)]
            for t in ts:
                t_to_k4[t].append(i)
                
        for i, k5 in enumerate(k5s):
            ts = [t_idx[t] for t in combinations(k5, 3)]
            for t in ts:
                t_to_k5[t].append(i)
                
        # Convert to numpy arrays for speed
        t_to_k4_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in t_to_k4]
        t_to_k5_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in t_to_k5]
        
        return triples, t_to_k4_np, t_to_k5_np, len(k4s), len(k5s)

    def solve_n(n, time_budget):
        if time.time() > deadline - 5:
            return None
            
        triples, t_to_k4_np, t_to_k5_np, num_k4, num_k5 = get_precomputed(n)
        num_t = len(triples)
        
        # Try multiple restarts
        num_restarts = 5
        time_per_restart = time_budget / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 2:
                break
                
            t_end = time.time() + time_per_restart
            
            # Random Init
            colors = np.random.randint(0, 2, size=num_t)
            
            # Init Counts
            red_count_k4 = np.zeros(num_k4, dtype=np.int32)
            blue_count_k5 = np.zeros(num_k5, dtype=np.int32)
            
            # This init loop is slow in Python. Vectorize?
            # For each triple, if 0, add to red_count_k4 for its k4s.
            # Hard to vectorize easily without sparse matrices.
            # num_t is ~ 9880 for n=40.
            # Loop is 9880 * (avg k4s) ~ 9880 * 37 ~ 365k ops. Fast enough.
            
            for t in range(num_t):
                if colors[t] == 0:
                    idxs = t_to_k4_np[t]
                    if len(idxs) > 0:
                        red_count_k4[idxs] += 1
                else:
                    idxs = t_to_k5_np[t]
                    if len(idxs) > 0:
                        blue_count_k5[idxs] += 1
                        
            viol_k4 = np.sum(red_count_k4 == 4)
            viol_k5 = np.sum(blue_count_k5 == 10)
            total_viol = viol_k4 + viol_k5
            
            if total_viol == 0:
                return colors.tolist()
                
            best_viol = total_viol
            best_colors = colors.copy()
            
            T = 1.0
            T_min = 0.01
            cooling = 0.9999
            steps = 0
            max_steps = 200000 # NumPy is fast, allow more steps
            
            while steps < max_steps and time.time() < t_end and time.time() < deadline - 2:
                t = random.randint(0, num_t - 1)
                old_c = colors[t]
                
                delta_viol = 0
                
                # K4
                idxs = t_to_k4_np[t]
                if len(idxs) > 0:
                    counts = red_count_k4[idxs]
                    if old_c == 0: # R -> B
                        delta_viol -= np.sum(counts == 4)
                    else: # B -> R
                        delta_viol += np.sum(counts == 3)
                        
                # K5
                idxs = t_to_k5_np[t]
                if len(idxs) > 0:
                    counts = blue_count_k5[idxs]
                    if old_c == 0: # R -> B
                        delta_viol += np.sum(counts == 9)
                    else: # B -> R
                        delta_viol -= np.sum(counts == 10)
                        
                # Accept
                if delta_viol < 0 or (T > T_min and random.random() < math.exp(-delta_viol / T)):
                    colors[t] = 1 - old_c
                    total_viol += delta_viol
                    
                    # Update counts
                    idxs = t_to_k4_np[t]
                    if len(idxs) > 0:
                        if old_c == 0:
                            red_count_k4[idxs] -= 1
                        else:
                            red_count_k4[idxs] += 1
                            
                    idxs = t_to_k5_np[t]
                    if len(idxs) > 0:
                        if old_c == 0:
                            blue_count_k5[idxs] += 1
                        else:
                            blue_count_k5[idxs] -= 1
                            
                    if total_viol < best_viol:
                        best_viol = total_viol
                        best_colors = colors.copy()
                        if best_viol == 0:
                            return best_colors.tolist()
                            
                T *= cooling
                steps += 1
                
        return None

    # Try n from 40 down to 34
    # Allocate time based on n
    # 40 is hardest. Give it more time?
    # Total 140s.
    # 40: 40s, 39: 20s, 38: 15s, 37: 10s, 36: 10s, 35: 10s, 34: 15s (fallback)
    
    time_alloc = {
        40: 40.0,
        39: 20.0,
        38: 15.0,
        37: 10.0,
        36: 10.0,
        35: 10.0,
        34: 15.0
    }
    
    for n in range(40, 33, -1):
        if time.time() > deadline - 5:
            break
        budget = time_alloc.get(n, 10.0)
        # Adjust budget if we are running low
        if time.time() + budget > deadline - 5:
            budget = deadline - 5 - time.time()
            if budget < 2:
                continue
                
        result = solve_n(n, budget)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    n = 34
    result = solve_n(n, deadline - 2 - time.time())
    if result is not None:
        return {'n': n, 'colors': result}
        
    # Last resort
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
