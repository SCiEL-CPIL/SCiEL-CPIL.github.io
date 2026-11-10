"""Rewritten program with Enhanced Algebraic Initialization and Efficient Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    def build_precomputation(n):
        """
        Precomputes indices for fast violation counting.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        # Map each K4 to its 4 constituent triple indices
        k4_triples = []
        for k4 in k4_list:
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
        k4_arr = np.array(k4_triples, dtype=np.int32)
        
        # Map each K5 to its 10 constituent triple indices
        k5_triples = []
        for k5 in k5_list:
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
        k5_arr = np.array(k5_triples, dtype=np.int32)
        
        # Inverse mapping: for each triple, which K4s and K5s does it belong to?
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k4_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4s[t_idx].append(k4_idx)
        
        triple_to_k5s = [[] for _ in range(num_triples)]
        for k5_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5s[t_idx].append(k5_idx)
                
        return triples, num_triples, k4_arr, k5_arr, triple_to_k4s, triple_to_k5s

    def run_local_search(n, time_budget, seed_offset=0):
        """
        Runs Local Search with Enhanced Algebraic Initialization.
        """
        triples, num_triples, k4_arr, k5_arr, triple_to_k4s, triple_to_k5s = build_precomputation(n)
        
        best_v = float('inf')
        best_colors = None
        
        # Generate a diverse set of initializations
        # We will evaluate each, and if it's promising (low violations), run local search from it.
        
        init_candidates = []
        
        # 1. Random starts
        for _ in range(5):
            init_candidates.append(('random', None))
            
        # 2. Modular patterns: (a+b+c) % m < k
        for m in [3, 5, 7, 11, 13, 17]:
            for k in range(1, m):
                # Only include if density is roughly balanced or biased towards red (since K4 is smaller)
                # Actually, let's just try a few.
                if m == 3:
                    if k in [1, 2]: init_candidates.append((f'mod{m}_{k}', (m, k)))
                elif m == 5:
                    if k in [2, 3]: init_candidates.append((f'mod{m}_{k}', (m, k)))
                elif m == 7:
                    if k in [3, 4]: init_candidates.append((f'mod{m}_{k}', (m, k)))
                else:
                    init_candidates.append((f'mod{m}_{k}', (m, k)))

        # 3. Algebraic patterns: Quadratic forms
        # Color = 0 if (a-b)^2 + (b-c)^2 + (c-a)^2 < threshold
        # Or based on product: a*b + b*c + c*a % m
        for m in [3, 5, 7, 11]:
            for k in range(1, m):
                init_candidates.append((f'alg_prod_{m}_{k}', ('prod', m, k)))
                init_candidates.append((f'alg_diff_{m}_{k}', ('diff', m, k)))

        # Shuffle candidates to ensure variety
        random.shuffle(init_candidates)
        
        # We will try to find a start with low violations first, then search.
        # To save time, we limit the number of initializations we fully evaluate and search.
        
        # Phase 1: Find the best starting point
        best_start_colors = None
        best_start_v = float('inf')
        best_start_k4 = None
        best_start_k5 = None
        
        # Evaluate first 20 candidates for violation count
        for name, params in init_candidates[:20]:
            if time.time() > start_time + time_budget * 0.1:
                break
                
            colors = np.zeros(num_triples, dtype=np.int8)
            
            if name == 'random':
                colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            else:
                if name.startswith('mod'):
                    m, k = params
                    for i, (a, b, c) in enumerate(triples):
                        if (a + b + c) % m < k:
                            colors[i] = 0
                        else:
                            colors[i] = 1
                elif name.startswith('alg_prod'):
                    _, m, k = params
                    for i, (a, b, c) in enumerate(triples):
                        # (a*b + b*c + c*a) % m < k
                        if (a*b + b*c + c*a) % m < k:
                            colors[i] = 0
                        else:
                            colors[i] = 1
                elif name.startswith('alg_diff'):
                    _, m, k = params
                    for i, (a, b, c) in enumerate(triples):
                        # ((a-b)^2 + (b-c)^2 + (c-a)^2) % m < k
                        d1 = (a-b)
                        d2 = (b-c)
                        d3 = (c-a)
                        val = (d1*d1 + d2*d2 + d3*d3) % m
                        if val < k:
                            colors[i] = 0
                        else:
                            colors[i] = 1
            
            # Compute violations
            k4_red_counts = np.sum(1 - colors[k4_arr], axis=1)
            k5_blue_counts = np.sum(colors[k5_arr], axis=1)
            
            num_red_violations = np.sum(k4_red_counts == 0)
            num_blue_violations = np.sum(k5_blue_counts == 10)
            current_v = num_red_violations + num_blue_violations
            
            if current_v < best_start_v:
                best_start_v = current_v
                best_start_colors = colors.copy()
                best_start_k4 = k4_red_counts.copy()
                best_start_k5 = k5_blue_counts.copy()
                
                if best_start_v == 0:
                    return n, best_start_colors.tolist()
        
        # Phase 2: Run local search from the best start
        if best_start_colors is not None:
            cur_colors = best_start_colors.copy()
            cur_k4_red = best_start_k4.copy()
            cur_k5_blue = best_start_k5.copy()
            cur_v = best_start_v
            
            T = 10.0
            T_end = 0.1
            cooling_rate = 0.9995
            no_improve = 0
            max_no_improve = 10000
            restart_count = 0
            max_restarts = 5
            
            while time.time() < start_time + time_budget:
                idx = random.randint(0, num_triples - 1)
                old_val = cur_colors[idx]
                new_val = 1 - old_val
                
                delta_v = 0
                
                # Update K4s
                for k4_idx in triple_to_k4s[idx]:
                    old_count = cur_k4_red[k4_idx]
                    if old_val == 0:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                    
                    was_violation = (old_count == 0)
                    is_violation = (new_count == 0)
                    
                    if was_violation and not is_violation:
                        delta_v -= 1
                    elif not was_violation and is_violation:
                        delta_v += 1
                        
                    cur_k4_red[k4_idx] = new_count
                    
                # Update K5s
                for k5_idx in triple_to_k5s[idx]:
                    old_count = cur_k5_blue[k5_idx]
                    if old_val == 1:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                    
                    was_violation = (old_count == 10)
                    is_violation = (new_count == 10)
                    
                    if was_violation and not is_violation:
                        delta_v -= 1
                    elif not was_violation and is_violation:
                        delta_v += 1
                        
                    cur_k5_blue[k5_idx] = new_count
                    
                new_v = cur_v + delta_v
                
                if new_v < cur_v:
                    cur_v = new_v
                    cur_colors[idx] = new_val
                    no_improve = 0
                    
                    if cur_v < best_v:
                        best_v = cur_v
                        best_colors = cur_colors.copy()
                        
                        if best_v == 0:
                            return n, best_colors.tolist()
                else:
                    if delta_v > 0:
                        if T > T_end:
                            prob = np.exp(-delta_v / T)
                            if random.random() < prob:
                                cur_v = new_v
                                cur_colors[idx] = new_val
                    # delta_v == 0: accept
                    else:
                        cur_v = new_v
                        cur_colors[idx] = new_val
                    no_improve += 1
                    
                T *= cooling_rate
                
                if no_improve > max_no_improve:
                    # Restart temperature
                    T = 10.0
                    no_improve = 0
                    restart_count += 1
                    if restart_count > max_restarts:
                        break
        
        if best_v == 0 and best_colors is not None:
            return n, best_colors.tolist()
        else:
            return None, None

    # Strategy:
    # 1. Try n=34 with the enhanced local search.
    # 2. If fail, try n=33.
    
    time_budget_34 = 120.0
    res_n, res_colors = run_local_search(34, time_budget_34)
    if res_n is not None:
        return {'n': res_n, 'colors': res_colors}
        
    # If n=34 failed, try n=33 with remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        res_n, res_colors = run_local_search(33, remaining - 5)
        if res_n is not None:
            return {'n': res_n, 'colors': res_colors}
            
    # Fallback: n=9
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

# EVOLVE-BLOCK-END
