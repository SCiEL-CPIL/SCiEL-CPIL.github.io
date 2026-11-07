import time
import random
from itertools import combinations
import math

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _soft_max(a, b, k=0.5):
    """Smooth max(a,b) using log-sum-exp."""
    if a == b:
        return a
    m = max(a, b)
    if m == 0:
        return 0
    return m + math.log(math.exp((a-m)/k) + math.exp((b-m)/k)) * k

def _fast_edge_optimize(values, m=35, deadline=None, num_restarts=15):
    """Fast edge optimization using bitmask-based collision tracking."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    edge_sums = [0] * num_all
    edge_prods = [0] * num_all
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = values[i] + values[j]
        edge_prods[idx] = values[i] * values[j]
    
    # Map sums and products to bit indices
    all_sums = sorted(set(edge_sums))
    all_prods = sorted(set(edge_prods))
    sum_to_bit = {s: i for i, s in enumerate(all_sums)}
    prod_to_bit = {p: i for i, p in enumerate(all_prods)}
    
    edge_sum_bits = [1 << sum_to_bit[edge_sums[idx]] for idx in range(num_all)]
    edge_prod_bits = [1 << prod_to_bit[edge_prods[idx]] for idx in range(num_all)]
    
    best_score = float('inf')
    best_edges = None
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Greedy init: pick edges that maximize collisions
        current = set()
        cur_sum_mask = 0
        cur_prod_mask = 0
        candidates = list(range(num_all))
        random.shuffle(candidates)
        
        for _ in range(m):
            best_idx = -1
            best_key = None
            for idx in candidates:
                if idx in current:
                    continue
                new_sum_mask = cur_sum_mask | edge_sum_bits[idx]
                new_prod_mask = cur_prod_mask | edge_prod_bits[idx]
                ns = new_sum_mask.bit_count()
                np_ = new_prod_mask.bit_count()
                # Use soft max for smoother gradient
                key = (_soft_max(ns, np_), ns + np_)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            if best_idx == -1:
                break
            current.add(best_idx)
            cur_sum_mask |= edge_sum_bits[best_idx]
            cur_prod_mask |= edge_prod_bits[best_idx]
            candidates.remove(best_idx)
        
        if len(current) < m:
            continue
        
        cur_score = max(cur_sum_mask.bit_count(), cur_prod_mask.bit_count())
        
        # Local search with swaps
        improved = True
        max_iter = 100
        it = 0
        while improved and it < max_iter and (not deadline or time.time() < deadline):
            improved = False
            it += 1
            cur_list = list(current)
            outside = [x for x in range(num_all) if x not in current]
            random.shuffle(cur_list)
            random.shuffle(outside)
            
            sum_val_counts = {}
            prod_val_counts = {}
            for idx in current:
                sv = edge_sums[idx]
                pv = edge_prods[idx]
                sum_val_counts[sv] = sum_val_counts.get(sv, 0) + 1
                prod_val_counts[pv] = prod_val_counts.get(pv, 0) + 1

            for e_in in cur_list:
                if deadline and time.time() > deadline:
                    break
                
                s_in_val = edge_sums[e_in]
                p_in_val = edge_prods[e_in]
                s_in_bit = edge_sum_bits[e_in]
                p_in_bit = edge_prod_bits[e_in]
                
                sum_val_counts[s_in_val] -= 1
                prod_val_counts[p_in_val] -= 1
                
                trial_sum_mask = cur_sum_mask
                trial_prod_mask = cur_prod_mask
                if sum_val_counts[s_in_val] == 0:
                    trial_sum_mask &= ~s_in_bit
                if prod_val_counts[p_in_val] == 0:
                    trial_prod_mask &= ~p_in_bit
                
                best_swap_found = False
                for e_out in outside:
                    s_out_val = edge_sums[e_out]
                    p_out_val = edge_prods[e_out]
                    s_out_bit = edge_sum_bits[e_out]
                    p_out_bit = edge_prod_bits[e_out]
                    
                    new_trial_sum_mask = trial_sum_mask
                    new_trial_prod_mask = trial_prod_mask
                    
                    if sum_val_counts.get(s_out_val, 0) > 0:
                        pass
                    else:
                        new_trial_sum_mask |= s_out_bit
                        
                    if prod_val_counts.get(p_out_val, 0) > 0:
                        pass
                    else:
                        new_trial_prod_mask |= p_out_bit
                    
                    trial_score = max(new_trial_sum_mask.bit_count(), new_trial_prod_mask.bit_count())
                    
                    if trial_score < cur_score:
                        current.discard(e_in)
                        current.add(e_out)
                        
                        sum_val_counts[s_out_val] = sum_val_counts.get(s_out_val, 0) + 1
                        prod_val_counts[p_out_val] = prod_val_counts.get(p_out_val, 0) + 1
                        
                        cur_sum_mask = new_trial_sum_mask
                        cur_prod_mask = new_trial_prod_mask
                        cur_score = trial_score
                        improved = True
                        best_swap_found = True
                        
                        if cur_score < best_score:
                            best_score = cur_score
                            best_edges = [list(all_edges[idx]) for idx in current]
                        
                        break
                
                if best_swap_found:
                    break
            
        if cur_score < best_score:
            best_score = cur_score
            best_edges = [list(all_edges[idx]) for idx in current]
    
    return best_edges, best_score

def _targeted_collision_move(vals, edges, deadline):
    """Try targeted moves that set a value to create sum/product collisions."""
    n = len(vals)
    best_improvement = None
    best_new_vals = None
    best_new_raw = float('inf')
    
    # Collect existing sums and products
    existing_sums = set()
    existing_prods = set()
    for i, j in edges:
        existing_sums.add(vals[i] + vals[j])
        existing_prods.add(vals[i] * vals[j])
        
    for vi in range(n):
        if deadline and time.time() > deadline:
            break
        neighbors = set()
        for i, j in edges:
            if i == vi:
                neighbors.add(j)
            elif j == vi:
                neighbors.add(i)
        
        candidates = set()
        for nj in neighbors:
            vj = vals[nj]
            # Sum collision
            for s in existing_sums:
                cand = s - vj
                if 1 <= cand <= 10000 and cand != vals[vi] and cand not in vals:
                    candidates.add(cand)
            # Product collision
            for p in existing_prods:
                if p % vj == 0:
                    cand = p // vj
                    if 1 <= cand <= 10000 and cand != vals[vi] and cand not in vals:
                        candidates.add(cand)
            
            # Also try values that are factors/multiples of vj
            for k in range(2, 11):
                if vj * k <= 10000:
                    candidates.add(vj * k)
                if vj % k == 0:
                    candidates.add(vj // k)
        
        # Small perturbations
        for delta in range(-5, 6):
            cand = vals[vi] + delta
            if 1 <= cand <= 10000 and cand not in vals:
                candidates.add(cand)
        
        for cand in list(candidates)[:30]:
            if deadline and time.time() > deadline:
                break
            trial_vals = list(vals)
            trial_vals[vi] = cand
            if len(set(trial_vals)) != 14:
                continue
            
            sums = set()
            prods = set()
            for i, j in edges:
                sums.add(trial_vals[i] + trial_vals[j])
                prods.add(trial_vals[i] * trial_vals[j])
            trial_raw = max(len(sums), len(prods))
            
            if trial_raw < best_new_raw:
                best_new_raw = trial_raw
                best_new_vals = trial_vals
    
    return best_new_vals, best_new_raw

def _polish(values, edges, deadline, max_rounds=5):
    """Dedicated polish phase with alternating edge and value local search."""
    vals = list(values)
    edg = [list(e) for e in edges]
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        # Edge polish
        new_edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=15)
        if new_edges and raw < _raw_value(vals, edg):
            edg = new_edges
        
        # Value polish
        new_vals, raw2 = _targeted_collision_move(vals, edg, deadline)
        if new_vals and raw2 < _raw_value(vals, edg):
            vals = new_vals
            new_edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=10)
            if new_edges:
                edg = new_edges
        
        # Random value perturbation
        trial_vals = list(vals)
        idxs = random.sample(range(14), random.randint(1, 3))
        for idx in idxs:
            delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            new_v = trial_vals[idx] + delta
            if 1 <= new_v <= 10000 and new_v not in trial_vals:
                trial_vals[idx] = new_v
        if len(set(trial_vals)) == 14:
            t_edges, t_raw = _fast_edge_optimize(trial_vals, 35, deadline, num_restarts=5)
            if t_edges and t_raw < _raw_value(vals, edg):
                vals = trial_vals
                edg = t_edges
    
    return {'values': vals, 'edges': edg}, _raw_value(vals, edg)

def _joint_optimize(start_values, deadline, max_rounds=8):
    """Jointly optimize values and edges through iterative perturbation with polish."""
    best_raw = float('inf')
    best_vals = None
    best_edges = None
    
    vals = list(start_values)
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=10)
        
        if edges and raw < best_raw:
            best_raw = raw
            best_vals = list(vals)
            best_edges = [list(e) for e in edges]
        
        new_vals, t_raw = _targeted_collision_move(vals, edges if edges else [], deadline)
        if new_vals and t_raw < best_raw:
            best_raw = t_raw
            best_vals = list(new_vals)
            t_edges, t_raw2 = _fast_edge_optimize(new_vals, 35, deadline, num_restarts=8)
            if t_edges and t_raw2 < best_raw:
                best_raw = t_raw2
                best_edges = [list(e) for e in t_edges]
            else:
                best_edges = [list(e) for e in edges] if edges else best_edges
            vals = new_vals
        elif edges:
            vals = best_vals if best_vals else vals
        
        # Random perturbation
        for _ in range(2):
            if deadline and time.time() > deadline:
                break
            trial_vals = list(vals)
            idxs = random.sample(range(14), random.randint(1, 3))
            for idx in idxs:
                delta = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
                new_v = trial_vals[idx] + delta
                if 1 <= new_v <= 10000 and new_v not in trial_vals:
                    trial_vals[idx] = new_v
            
            if len(set(trial_vals)) != 14:
                continue
            
            t_edges, t_raw = _fast_edge_optimize(trial_vals, 35, deadline, num_restarts=5)
            if t_edges and t_raw < best_raw:
                best_raw = t_raw
                best_vals = list(trial_vals)
                best_edges = [list(e) for e in t_edges]
                vals = trial_vals
    
    if best_vals and deadline and time.time() < deadline:
        polished, pol_raw = _polish(best_vals, best_edges, deadline, max_rounds=4)
        if pol_raw < best_raw:
            best_raw = pol_raw
            best_vals = polished['values']
            best_edges = polished['edges']
    
    if best_vals is None:
        return None, float('inf')
    return {'values': best_vals, 'edges': best_edges}, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_result = None
    best_raw = float('inf')
    
    value_sets = [
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 30, 36, 48, 60],
        [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 144],
        # Add a few more diverse sets
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43], # Primes (bad for products, good for sums?)
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27], # Odds
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196], # Squares
        # New structured sets for better collision potential
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610], # Fibonacci
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # Consecutive
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], # Duplicates (invalid but will be filtered)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15], # Mostly consecutive
    ]
    
    for vals in value_sets:
        if deadline and time.time() > deadline:
            break
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
        
        edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=12)
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    if best_result and deadline and time.time() < deadline:
        jr, jr_raw = _joint_optimize(best_result['values'], deadline, max_rounds=8)
        if jr and jr_raw < best_raw:
            best_raw = jr_raw
            best_result = jr
    
    if deadline and time.time() < deadline - 15:
        for vals in value_sets[1:5]:
            if deadline and time.time() > deadline:
                break
            jr, jr_raw = _joint_optimize(vals, deadline, max_rounds=5)
            if jr and jr_raw < best_raw:
                best_raw = jr_raw
                best_result = jr
    
    if best_result and deadline and time.time() < deadline - 5:
        polished, pol_raw = _polish(best_result['values'], best_result['edges'], deadline, max_rounds=5)
        if pol_raw < best_raw:
            best_raw = pol_raw
            best_result = polished
    
    if best_result is None:
        vals = list(range(1, 15))
        edges = list(combinations(range(14), 2))[:35]
        best_result = {'values': vals, 'edges': edges}
    
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
