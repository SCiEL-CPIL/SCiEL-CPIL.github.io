import time
import random
from itertools import combinations
from collections import Counter
import math

DEADLINE = 88.0  # Safety margin for 90s limit

def _get_distinct_counts(values, edges):
    """Calculate distinct sums and products for a given set of edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return len(sums), len(prods)

def _optimize_edges(values, m, deadline):
    """
    Optimize the set of m edges for the given values using Simulated Annealing.
    Returns (best_edges, best_score).
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [values[i] + values[j] for i, j in all_edges]
    edge_prods = [values[i] * values[j] for i, j in all_edges]
    
    # Index mapping for faster lookups in counters? 
    # Actually, Counter with int keys is fast enough.
    
    best_score = float('inf')
    best_edges_set = None
    rng = random.Random(42)
    
    # Initialize with a greedy approach or random
    # Greedy: Add edges that introduce fewest new sums/products
    current_edges_set = set()
    sum_counts = Counter()
    prod_counts = Counter()
    remaining = list(range(num_edges))
    rng.shuffle(remaining)
    
    # Greedy initialization
    for _ in range(m):
        best_idx = -1
        best_cost = float('inf')
        # Sample a subset of remaining edges to speed up greedy init if num_edges is large
        # But 91 is small, so full scan is fine.
        for idx in remaining:
            s = edge_sums[idx]
            p = edge_prods[idx]
            
            old_s = sum_counts[s]
            old_p = prod_counts[p]
            
            # Simulate add
            sum_counts[s] = old_s + 1
            prod_counts[p] = old_p + 1
            
            # Cost is max(distinct sums, distinct prods)
            # We don't need exact count if we just want to minimize marginal increase?
            # No, we need the actual score.
            # To speed up, we can just track the counts of new unique items.
            # But calculating max(len(sums), len(prods)) is O(1) if we maintain len.
            # Let's just calculate the score.
            # This is O(1) if we maintain the sets, but we are using Counters.
            # We can maintain distinct_sums and distinct_prods counts.
            
            # Revert for now, let's use a simpler cost function for greedy:
            # Prefer edges where sum and product are already frequent.
            cost = -(old_s + old_p) # Higher is better (more collisions)
            
            # Revert
            if old_s == 0:
                del sum_counts[s]
            else:
                sum_counts[s] = old_s
            if old_p == 0:
                del prod_counts[p]
            else:
                prod_counts[p] = old_p
            
            if cost > best_cost:
                best_cost = cost
                best_idx = idx
        
        if best_idx != -1:
            s = edge_sums[best_idx]
            p = edge_prods[best_idx]
            sum_counts[s] += 1
            prod_counts[p] += 1
            current_edges_set.add(best_idx)
            remaining.remove(best_idx)
            
    current_score = max(len(sum_counts), len(prod_counts))
    if current_score < best_score:
        best_score = current_score
        best_edges_set = set(current_edges_set)
        
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    iters = 0
    max_iters = 20000
    no_improve = 0
    patience = 2000
    
    while iters < max_iters:
        if time.time() > deadline:
            break
            
        # Pick a random edge in and a random edge out
        if not current_edges_set:
            break
        in_list = list(current_edges_set)
        out_list = [i for i in range(num_edges) if i not in current_edges_set]
        if not out_list:
            break
            
        idx_in = rng.choice(in_list)
        idx_out = rng.choice(out_list)
        
        s_in = edge_sums[idx_in]
        p_in = edge_prods[idx_in]
        s_out = edge_sums[idx_out]
        p_out = edge_prods[idx_out]
        
        # Update counters
        old_s_in = sum_counts[s_in]
        old_p_in = prod_counts[p_in]
        old_s_out = sum_counts[s_out]
        old_p_out = prod_counts[p_out]
        
        # Remove in
        if old_s_in == 1:
            del sum_counts[s_in]
        else:
            sum_counts[s_in] = old_s_in - 1
        if old_p_in == 1:
            del prod_counts[p_in]
        else:
            prod_counts[p_in] = old_p_in - 1
            
        # Add out
        sum_counts[s_out] = old_s_out + 1
        prod_counts[p_out] = old_p_out + 1
        
        new_score = max(len(sum_counts), len(prod_counts))
        
        accept = False
        if new_score < current_score:
            accept = True
        elif new_score == current_score:
            accept = True
        else:
            delta = new_score - current_score
            accept = rng.random() < math.exp(-delta / max(T, 1e-10))
            
        if accept:
            current_score = new_score
            current_edges_set.remove(idx_in)
            current_edges_set.add(idx_out)
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_edges_set = set(current_edges_set)
        else:
            # Revert
            if old_s_out == 0:
                del sum_counts[s_out]
            else:
                sum_counts[s_out] = old_s_out
            if old_p_out == 0:
                del prod_counts[p_out]
            else:
                prod_counts[p_out] = old_p_out
            sum_counts[s_in] = old_s_in
            prod_counts[p_in] = old_p_in
            no_improve += 1
            
        if no_improve > patience:
            break
            
        T *= alpha
        if T < T_min:
            T = T_min
        iters += 1
        
    # Polish: Local search for 1-swap improvements
    if best_edges_set and time.time() < deadline:
        current_edges_set = set(best_edges_set)
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in current_edges_set:
            sum_counts[edge_sums[idx]] += 1
            prod_counts[edge_prods[idx]] += 1
        current_score = max(len(sum_counts), len(prod_counts))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            in_list = list(current_edges_set)
            out_list = [i for i in range(num_edges) if i not in current_edges_set]
            
            for idx_in in in_list:
                for idx_out in out_list:
                    s_in = edge_sums[idx_in]
                    p_in = edge_prods[idx_in]
                    s_out = edge_sums[idx_out]
                    p_out = edge_prods[idx_out]
                    
                    old_s_in = sum_counts[s_in]
                    old_p_in = prod_counts[p_in]
                    old_s_out = sum_counts[s_out]
                    old_p_out = prod_counts[p_out]
                    
                    # Remove in
                    if old_s_in == 1:
                        del sum_counts[s_in]
                    else:
                        sum_counts[s_in] = old_s_in - 1
                    if old_p_in == 1:
                        del prod_counts[p_in]
                    else:
                        prod_counts[p_in] = old_p_in - 1
                        
                    # Add out
                    sum_counts[s_out] = old_s_out + 1
                    prod_counts[p_out] = old_p_out + 1
                    
                    new_score = max(len(sum_counts), len(prod_counts))
                    
                    if new_score < current_score:
                        current_score = new_score
                        current_edges_set.remove(idx_in)
                        current_edges_set.add(idx_out)
                        best_score = new_score
                        best_edges_set = set(current_edges_set)
                        improved = True
                        break
                    else:
                        # Revert
                        if old_s_out == 0:
                            del sum_counts[s_out]
                        else:
                            sum_counts[s_out] = old_s_out
                        if old_p_out == 0:
                            del prod_counts[p_out]
                        else:
                            prod_counts[p_out] = old_p_out
                        sum_counts[s_in] = old_s_in
                        prod_counts[p_in] = old_p_in
                        
                if improved:
                    break
                    
    if best_edges_set is None:
        best_edges_set = set(range(m))
        
    return [all_edges[idx] for idx in best_edges_set], best_score

def construct():
    start_time = time.time()
    n = 14
    m = 35
    
    # Generate candidate value sets
    # 1. Structured sets
    candidates = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 24],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 25, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 18, 20, 24, 27],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 25],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 40],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 25, 30],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48, 72, 96],
    ]
    
    # 2. Random sets with bias towards small numbers (more factors)
    rng = random.Random(123)
    for _ in range(20):
        vals = []
        # Generate from 1..50
        while len(vals) < n:
            v = rng.randint(1, 50)
            if v not in vals:
                vals.append(v)
        candidates.append(vals)
        
    # 3. Random sets from 1..100
    for _ in range(10):
        vals = rng.sample(range(1, 101), n)
        candidates.append(vals)
        
    # 4. Smooth numbers sets
    for _ in range(10):
        vals = set()
        while len(vals) < n:
            v = 1
            # Random exponents for 2, 3, 5, 7
            e2 = rng.randint(0, 4)
            e3 = rng.randint(0, 3)
            e5 = rng.randint(0, 2)
            e7 = rng.randint(0, 2)
            v = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7)
            if v <= 10000:
                vals.add(v)
        if len(vals) >= n:
            candidates.append(list(vals)[:n])
            
    # Deduplicate and validate candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        if len(c) != n or len(set(c)) != n:
            continue
        if any(v < 1 or v > 10000 for v in c):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Sort candidates by a quick heuristic: number of distinct products in all edges
    # This is expensive, so do it only for top few or use a proxy.
    # Proxy: sum of log(v) ? No.
    # Let's just run the optimizer on the first few and the structured ones.
    
    # We will evaluate candidates in batches.
    # Time budget: 30s for value selection (quick eval), 55s for deep edge opt.
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Quick evaluation of all candidates with limited time per candidate
    quick_deadline = start_time + 25.0
    for values in unique_candidates:
        if time.time() > quick_deadline:
            break
            
        # Quick edge optimization: just greedy + short SA
        deadline_for_this = min(quick_deadline, time.time() + 1.5)
        try:
            edges, score = _optimize_edges(values, m, deadline_for_this)
        except Exception:
            continue
            
        if score is not None and score < best_score:
            best_score = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }
            
    # Phase 2: Deep optimization on the best few candidates
    # Identify top candidates from Phase 1? 
    # We didn't store scores for all, only the best.
    # Let's re-evaluate the top 5 structured ones and the best found so far.
    
    # To be safe, we'll just take the best_result and re-optimize it with more time.
    # And also try a few other promising ones.
    
    if best_result is not None and time.time() < DEADLINE - 5:
        deadline_for_deep = DEADLINE - 2
        try:
            edges, score = _optimize_edges(best_result['values'], m, deadline_for_deep)
            if score is not None and score < best_score:
                best_score = score
                best_result = {
                    'values': [int(v) for v in best_result['values']],
                    'edges': [[int(i), int(j)] for i, j in edges]
                }
        except Exception:
            pass
            
    # Also try one more random set if time permits
    if time.time() < DEADLINE - 10:
        vals = rng.sample(range(1, 101), n)
        deadline_for_deep = DEADLINE - 2
        try:
            edges, score = _optimize_edges(vals, m, deadline_for_deep)
            if score is not None and score < best_score:
                best_score = score
                best_result = {
                    'values': [int(v) for v in vals],
                    'edges': [[int(i), int(j)] for i, j in edges]
                }
        except Exception:
            pass

    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}

    return best_result
