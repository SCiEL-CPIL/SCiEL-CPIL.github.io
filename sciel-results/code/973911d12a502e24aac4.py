"""
Optimized program using Vectorized Simulated Annealing.
Avoids expensive orbit precomputation.
Uses direct vectorized evaluation of violations.
"""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Try n=34 first
    n = 34
    triples = list(combinations(range(n), 3))
    M = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4 and K5 edge indices
    k4_list = list(combinations(range(n), 4))
    k5_list = list(combinations(range(n), 5))
    
    N4 = len(k4_list)
    N5 = len(k5_list)
    
    k4_edges = np.zeros((N4, 4), dtype=np.int32)
    k5_edges = np.zeros((N5, 10), dtype=np.int32)
    
    # Precompute K4 edges
    for i, k4 in enumerate(k4_list):
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            k4_edges[i, j] = idx
            
    # Precompute K5 edges
    for i, k5 in enumerate(k5_list):
        for j, combo in enumerate(combinations(range(5), 3)):
            t = tuple(sorted([k5[x] for x in combo]))
            idx = triple_to_idx[t]
            k5_edges[i, j] = idx
            
    def get_violations(colors):
        # Red violations: K4 where all 4 are red (0)
        # Blue violations: K5 where all 10 are blue (1)
        
        # Check K4s
        red_counts = np.sum(1 - colors[k4_edges], axis=1)
        red_violations = np.sum(red_counts == 4)
        
        # Check K5s
        blue_counts = np.sum(colors[k5_edges], axis=1)
        blue_violations = np.sum(blue_counts == 10)
        
        return int(red_violations + blue_violations)

    def run_sa(n_vars, time_budget):
        # Initialize colors
        colors = np.random.randint(0, 2, size=n_vars, dtype=np.int8)
        
        cur_violations = get_violations(colors)
        best_violations = cur_violations
        best_colors = colors.copy()
        
        T = 10.0
        T_min = 0.1
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 2000
        kicks = 0
        max_kicks = 50
        
        while time.time() < start_time + time_budget:
            if cur_violations == 0:
                return colors.tolist()
            
            if cur_violations < best_violations:
                best_violations = cur_violations
                best_colors = colors.copy()
                no_improve = 0
                # T = max(T * 1.05, 10.0) # Reheat slightly
            else:
                no_improve += 1
                
            if no_improve > max_no_improve:
                kicks += 1
                if kicks > max_kicks:
                    break
                    
                # Kick: flip 10% of bits
                num_flips = max(1, n_vars // 10)
                flip_indices = np.random.choice(n_vars, num_flips, replace=False)
                colors[flip_indices] = 1 - colors[flip_indices]
                
                cur_violations = get_violations(colors)
                T = 10.0
                no_improve = 0
                continue
            
            # Single Flip
            idx = random.randint(0, n_vars - 1)
            old_val = colors[idx]
            new_val = 1 - old_val
            
            colors[idx] = new_val
            new_violations = get_violations(colors)
            
            if new_violations < cur_violations:
                cur_violations = new_violations
            else:
                if T > T_min:
                    delta = new_violations - cur_violations
                    if delta > 0:
                        if delta < 50 * T:
                            prob = np.exp(-delta / T)
                            if random.random() < prob:
                                cur_violations = new_violations
                            else:
                                colors[idx] = old_val
                        else:
                            colors[idx] = old_val
                    else:
                        cur_violations = new_violations
                else:
                    colors[idx] = old_val
                    
            T *= cooling
            
        if best_violations == 0:
            return best_colors.tolist()
        return None

    # Run SA for n=34
    res = run_sa(M, deadline - start_time - 5)
    
    if res is not None:
        return {'n': 34, 'colors': res}
    
    # Fallback: Try n=30 with a simpler, faster search or known pattern
    # If 34 fails, try 30
    n = 30
    triples = list(combinations(range(n), 3))
    M = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4_list = list(combinations(range(n), 4))
    k5_list = list(combinations(range(n), 5))
    
    N4 = len(k4_list)
    N5 = len(k5_list)
    
    k4_edges = np.zeros((N4, 4), dtype=np.int32)
    k5_edges = np.zeros((N5, 10), dtype=np.int32)
    
    for i, k4 in enumerate(k4_list):
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            k4_edges[i, j] = idx
            
    for i, k5 in enumerate(k5_list):
        for j, combo in enumerate(combinations(range(5), 3)):
            t = tuple(sorted([k5[x] for x in combo]))
            idx = triple_to_idx[t]
            k5_edges[i, j] = idx
            
    def get_violations_30(colors):
        red_counts = np.sum(1 - colors[k4_edges], axis=1)
        red_violations = np.sum(red_counts == 4)
        blue_counts = np.sum(colors[k5_edges], axis=1)
        blue_violations = np.sum(blue_counts == 10)
        return int(red_violations + blue_violations)

    def run_sa_30(n_vars, time_budget):
        colors = np.random.randint(0, 2, size=n_vars, dtype=np.int8)
        cur_violations = get_violations_30(colors)
        best_violations = cur_violations
        best_colors = colors.copy()
        
        T = 10.0
        T_min = 0.1
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 2000
        kicks = 0
        max_kicks = 20
        
        while time.time() < start_time + time_budget:
            if cur_violations == 0:
                return colors.tolist()
            
            if cur_violations < best_violations:
                best_violations = cur_violations
                best_colors = colors.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve > max_no_improve:
                kicks += 1
                if kicks > max_kicks:
                    break
                    
                num_flips = max(1, n_vars // 10)
                flip_indices = np.random.choice(n_vars, num_flips, replace=False)
                colors[flip_indices] = 1 - colors[flip_indices]
                
                cur_violations = get_violations_30(colors)
                T = 10.0
                no_improve = 0
                continue
            
            idx = random.randint(0, n_vars - 1)
            old_val = colors[idx]
            new_val = 1 - old_val
            
            colors[idx] = new_val
            new_violations = get_violations_30(colors)
            
            if new_violations < cur_violations:
                cur_violations = new_violations
            else:
                if T > T_min:
                    delta = new_violations - cur_violations
                    if delta > 0:
                        if delta < 50 * T:
                            prob = np.exp(-delta / T)
                            if random.random() < prob:
                                cur_violations = new_violations
                            else:
                                colors[idx] = old_val
                        else:
                            colors[idx] = old_val
                    else:
                        cur_violations = new_violations
                else:
                    colors[idx] = old_val
                    
            T *= cooling
            
        if best_violations == 0:
            return best_colors.tolist()
        return None

    res_30 = run_sa_30(M, deadline - start_time - 2)
    
    if res_30 is not None:
        return {'n': 30, 'colors': res_30}
        
    # Final Fallback: n=9
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
