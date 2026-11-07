import time
import random
from itertools import combinations

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

def _edge_score(vals, edge_list):
    """Fast score for a fixed set of edges."""
    sums = set()
    prods = set()
    for i, j in edge_list:
        sums.add(vals[i] + vals[j])
        prods.add(vals[i] * vals[j])
    return max(len(sums), len(prods))

def _greedy_edges(vals, m=35, deadline=None, num_restarts=20, seed=None):
    """Greedy edge selection with collision-aware scoring."""
    n = len(vals)
    rng = random.Random(seed)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [0] * num_all
    edge_prods = [0] * num_all
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = vals[i] + vals[j]
        edge_prods[idx] = vals[i] * vals[j]
    
    # Map to unique indices
    all_sums = sorted(set(edge_sums))
    all_prods = sorted(set(edge_prods))
    sum_to_idx = {s: i for i, s in enumerate(all_sums)}
    prod_to_idx = {p: i for i, p in enumerate(all_prods)}
    
    edge_sum_idx = [sum_to_idx[edge_sums[idx]] for idx in range(num_all)]
    edge_prod_idx = [prod_to_idx[edge_prods[idx]] for idx in range(num_all)]
    
    best_score = float('inf')
    best_edges = None
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Greedy construction
        current = set()
        sum_mask = 0
        prod_mask = 0
        candidates = list(range(num_all))
        rng.shuffle(candidates)
        
        for _ in range(m):
            best_idx = -1
            best_key = None
            for idx in candidates:
                if idx in current:
                    continue
                s_idx = edge_sum_idx[idx]
                p_idx = edge_prod_idx[idx]
                new_sum_mask = sum_mask | (1 << s_idx)
                new_prod_mask = prod_mask | (1 << p_idx)
                ns = new_sum_mask.bit_count()
                np_ = new_prod_mask.bit_count()
                # Prefer minimizing max, then sum
                key = (max(ns, np_), ns + np_)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            if best_idx == -1:
                break
            current.add(best_idx)
            sum_mask |= (1 << edge_sum_idx[best_idx])
            prod_mask |= (1 << edge_prod_idx[best_idx])
            candidates.remove(best_idx)
        
        if len(current) < m:
            continue
        
        cur_score = max(sum_mask.bit_count(), prod_mask.bit_count())
        
        # Local search: 1-swap moves
        improved = True
        max_iter = 200
        it = 0
        while improved and it < max_iter:
            if deadline and time.time() > deadline:
                break
            improved = False
            it += 1
            
            # Build count dicts
            sum_counts = {}
            prod_counts = {}
            for idx in current:
                si = edge_sum_idx[idx]
                pi = edge_prod_idx[idx]
                sum_counts[si] = sum_counts.get(si, 0) + 1
                prod_counts[pi] = prod_counts.get(pi, 0) + 1
            
            cur_list = list(current)
            outside = [x for x in range(num_all) if x not in current]
            rng.shuffle(cur_list)
            rng.shuffle(outside)
            
            for e_in in cur_list:
                if deadline and time.time() > deadline:
                    break
                si_in = edge_sum_idx[e_in]
                pi_in = edge_prod_idx[e_in]
                
                # Remove e_in
                sum_counts[si_in] -= 1
                prod_counts[pi_in] -= 1
                
                trial_sum_mask = sum_mask
                trial_prod_mask = prod_mask
                if sum_counts[si_in] == 0:
                    trial_sum_mask &= ~(1 << si_in)
                if prod_counts[pi_in] == 0:
                    trial_prod_mask &= ~(1 << pi_in)
                
                found_swap = False
                for e_out in outside:
                    si_out = edge_sum_idx[e_out]
                    pi_out = edge_prod_idx[e_out]
                    
                    new_trial_sum_mask = trial_sum_mask
                    new_trial_prod_mask = trial_prod_mask
                    
                    if sum_counts.get(si_out, 0) == 0:
                        new_trial_sum_mask |= (1 << si_out)
                    if prod_counts.get(pi_out, 0) == 0:
                        new_trial_prod_mask |= (1 << pi_out)
                    
                    trial_score = max(new_trial_sum_mask.bit_count(), new_trial_prod_mask.bit_count())
                    
                    if trial_score < cur_score:
                        # Apply swap
                        current.discard(e_in)
                        current.add(e_out)
                        sum_counts[si_out] = sum_counts.get(si_out, 0) + 1
                        prod_counts[pi_out] = prod_counts.get(pi_out, 0) + 1
                        sum_mask = new_trial_sum_mask
                        prod_mask = new_trial_prod_mask
                        cur_score = trial_score
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
        if cur_score < best_score:
            best_score = cur_score
            best_edges = [list(all_edges[idx]) for idx in current]
    
    return best_edges, best_score

def _value_optimization(vals, edges, deadline, max_rounds=5):
    """Optimize values given fixed edges."""
    vals = list(vals)
    n = len(vals)
    best_raw = _raw_value(vals, edges)
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        for vi in range(n):
            if deadline and time.time() > deadline:
                break
            
            # Find neighbors
            neighbors = set()
            for i, j in edges:
                if i == vi:
                    neighbors.add(j)
                elif j == vi:
                    neighbors.add(i)
            
            # Current sums and products involving vi
            cur_sums = set()
            cur_prods = set()
            for i, j in edges:
                if i == vi or j == vi:
                    cur_sums.add(vals[i] + vals[j])
                    cur_prods.add(vals[i] * vals[j])
            
            # All current sums and products
            all_sums = set()
            all_prods = set()
            for i, j in edges:
                all_sums.add(vals[i] + vals[j])
                all_prods.add(vals[i] * vals[j])
            
            # Candidate values
            candidates = set()
            
            # For sum collisions: try s - vj for each neighbor vj and target sum s
            for nj in neighbors:
                vj = vals[nj]
                for s in all_sums:
                    cand = s - vj
                    if 1 <= cand <= 10000 and cand != vals[vi] and cand not in vals:
                        candidates.add(cand)
                for p in all_prods:
                    if p % vj == 0:
                        cand = p // vj
                        if 1 <= cand <= 10000 and cand != vals[vi] and cand not in vals:
                            candidates.add(cand)
            
            # Small perturbations
            for delta in range(-10, 11):
                cand = vals[vi] + delta
                if 1 <= cand <= 10000 and cand not in vals:
                    candidates.add(cand)
            
            # Factor-based
            for k in range(2, 20):
                if vals[vi] * k <= 10000:
                    candidates.add(vals[vi] * k)
                if vals[vi] % k == 0:
                    candidates.add(vals[vi] // k)
            
            # Try a subset
            cand_list = list(candidates)[:50]
            random.shuffle(cand_list)
            
            for cand in cand_list:
                if deadline and time.time() > deadline:
                    break
                trial = list(vals)
                trial[vi] = cand
                if len(set(trial)) != 14:
                    continue
                trial_raw = _raw_value(trial, edges)
                if trial_raw < best_raw:
                    best_raw = trial_raw
                    vals = trial
                    improved = True
                    break
        
        if not improved:
            break
    
    return vals, best_raw

def _joint_search(start_vals, deadline, max_rounds=10):
    """Jointly optimize values and edges."""
    vals = list(start_vals)
    edges, raw = _greedy_edges(vals, 35, deadline, num_restarts=15)
    best_raw = raw
    best_vals = list(vals)
    best_edges = [list(e) for e in edges]
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        # Optimize values
        new_vals, val_raw = _value_optimization(vals, edges, deadline, max_rounds=3)
        if val_raw < best_raw:
            best_raw = val_raw
            best_vals = list(new_vals)
            vals = new_vals
            # Re-optimize edges with new values
            edges, edge_raw = _greedy_edges(vals, 35, deadline, num_restarts=10)
            if edge_raw < best_raw:
                best_raw = edge_raw
                best_edges = [list(e) for e in edges]
        
        # Random perturbation
        trial_vals = list(vals)
        num_perturb = random.randint(1, 3)
        for _ in range(num_perturb):
            idx = random.randint(0, 13)
            delta = random.choice([-10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
            new_v = trial_vals[idx] + delta
            if 1 <= new_v <= 10000 and new_v not in trial_vals:
                trial_vals[idx] = new_v
        
        if len(set(trial_vals)) == 14:
            t_edges, t_raw = _greedy_edges(trial_vals, 35, deadline, num_restarts=8)
            if t_edges and t_raw < best_raw:
                best_raw = t_raw
                best_vals = list(trial_vals)
                best_edges = [list(e) for e in t_edges]
                vals = trial_vals
    
    return {'values': best_vals, 'edges': best_edges}, best_raw

def _polish(values, edges, deadline, max_rounds=3):
    """Final polish with tighter search."""
    vals = list(values)
    edg = [list(e) for e in edges]
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        # Edge polish
        new_edges, raw = _greedy_edges(vals, 35, deadline, num_restarts=20)
        if new_edges and raw < _raw_value(vals, edg):
            edg = new_edges
        
        # Value polish
        new_vals, raw2 = _value_optimization(vals, edg, deadline, max_rounds=2)
        if raw2 < _raw_value(vals, edg):
            vals = new_vals
            new_edges, raw3 = _greedy_edges(vals, 35, deadline, num_restarts=15)
            if new_edges:
                edg = new_edges
    
    return {'values': vals, 'edges': edg}, _raw_value(vals, edg)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_result = None
    best_raw = float('inf')
    
    # Structured diverse initializations
    value_sets = [
        # Consecutive
        list(range(1, 15)),
        # Powers of 2
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Small highly composite
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 30, 36, 48, 60, 72],
        # Mixed small
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Fibonacci-like
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377],
        # Clusters
        [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23],
        # Divisors of 2520
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18],
        # Geometric
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Arithmetic
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        # Mixed
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
    ]
    
    # Deduplicate
    seen = set()
    unique_sets = []
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(vs)
    
    for vals in unique_sets:
        if deadline and time.time() > deadline:
            break
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
        
        edges, raw = _greedy_edges(vals, 35, deadline, num_restarts=15, seed=42)
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    if best_result and deadline and time.time() < deadline - 5:
        jr, jr_raw = _joint_search(best_result['values'], deadline, max_rounds=8)
        if jr and jr_raw < best_raw:
            best_raw = jr_raw
            best_result = jr
    
    if deadline and time.time() < deadline - 15:
        # Try a few more starting points
        for vals in unique_sets[1:4]:
            if deadline and time.time() > deadline:
                break
            jr, jr_raw = _joint_search(vals, deadline, max_rounds=5)
            if jr and jr_raw < best_raw:
                best_raw = jr_raw
                best_result = jr
    
    if best_result and deadline and time.time() < deadline - 3:
        polished, pol_raw = _polish(best_result['values'], best_result['edges'], deadline, max_rounds=3)
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
