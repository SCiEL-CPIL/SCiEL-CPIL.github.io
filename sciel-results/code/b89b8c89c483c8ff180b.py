import time
import random
import math
from itertools import combinations

_DEADLINE = time.time() + 80

def _raw_score(values, edge_pairs):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for (u, v) in edge_pairs:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    return list(combinations(range(n), 2))

def _greedy_select(sp, edge_list, m):
    """Greedy: pick edges that cause fewest new distinct sums/prods."""
    selected = []
    sums = set()
    prods = set()
    remaining = list(edge_list)
    
    # Sort by product frequency (prefer edges whose product is shared)
    prod_count = {}
    sum_count = {}
    for e in edge_list:
        s, p = sp[e]
        prod_count[p] = prod_count.get(p, 0) + 1
        sum_count[s] = sum_count.get(s, 0) + 1
        
    edge_list_sorted = sorted(edge_list, key=lambda e: (prod_count[sp[e][1]], sum_count[sp[e][0]]), reverse=True)
    
    for e in edge_list_sorted:
        if len(selected) >= m:
            break
        s, p = sp[e]
        selected.append(e)
        sums.add(s)
        prods.add(p)
        
    # Fill if needed (should not happen if m <= len(edge_list))
    for e in edge_list:
        if len(selected) >= m:
            break
        if e not in selected:
            selected.append(e)
            s, p = sp[e]
            sums.add(s)
            prods.add(p)
            
    return selected, sums, prods

def _local_search(values, sp, edge_list, m, max_iter, deadline):
    """Local search to minimize max(distinct sums, distinct products)."""
    n = len(values)
    all_edges = set(edge_list)
    
    # Start with greedy
    edges, sums, prods = _greedy_select(sp, edge_list, m)
    current_score = max(len(sums), len(prods))
    best_score = current_score
    best_edges = list(edges)
    
    edge_set = set(edges)
    
    # Precompute sum and product for all edges
    edge_sp = {e: sp[e] for e in all_edges}
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if current_score <= 2:
            break
        
        # Pick a random edge to remove
        u, v = random.choice(list(edge_set))
        s_old, p_old = edge_sp[(u, v)]
        
        # Compute counts
        s_counts = {}
        p_counts = {}
        for (x, y) in edge_set:
            sx, px = edge_sp[(x, y)]
            s_counts[sx] = s_counts.get(sx, 0) + 1
            p_counts[px] = p_counts.get(px, 0) + 1
            
        # Remove this edge
        s_counts[s_old] -= 1
        p_counts[p_old] -= 1
        if s_counts[s_old] == 0:
            del s_counts[s_old]
        if p_counts[p_old] == 0:
            del p_counts[p_old]
        
        # Try to add a different edge
        candidates = [e for e in all_edges if e not in edge_set]
        if not candidates:
            break
        
        # Prefer candidates with common sums/products
        best_add = None
        best_new_score = current_score + 1
        
        # Sample a subset of candidates for speed
        sample_size = min(50, len(candidates))
        sample = random.sample(candidates, sample_size)
        
        for cand in sample:
            sx, px = edge_sp[cand]
            
            new_s_count = len(s_counts) + (1 if sx not in s_counts else 0)
            new_p_count = len(p_counts) + (1 if px not in p_counts else 0)
            new_score = max(new_s_count, new_p_count)
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_add = cand
        
        if best_add is not None and best_new_score <= current_score:
            edge_set.discard((u, v))
            edge_set.add(best_add)
            sx, px = edge_sp[best_add]
            s_counts[sx] = s_counts.get(sx, 0) + 1
            p_counts[px] = p_counts.get(px, 0) + 1
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edge_set)
                
    return best_score, best_edges

def construct():
    n = 14
    m = 35
    max_val = 10000
    deadline = time.time() + 78
    
    candidates = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 30],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
        [1, 2, 3, 4, 6, 8, 9, 12, 14, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 9, 10, 12, 14, 15, 16, 18, 20, 25],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20],
        # Add divisors of 2520 (highly composite)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 18, 21],
    ]
    
    valid_candidates = []
    seen = set()
    for vals in candidates:
        if len(set(vals)) != n:
            continue
        if min(vals) < 1 or max(vals) > max_val:
            continue
        key = tuple(sorted(vals))
        if key in seen:
            continue
        seen.add(key)
        valid_candidates.append(list(vals))
        
    random.seed(123)
    for _ in range(50):
        vals = sorted(random.sample(range(1, 50), n))
        key = tuple(vals)
        if key not in seen:
            seen.add(key)
            valid_candidates.append(vals)
            
    best_score = m
    best_result = None
    edge_list = _all_edges(n)
    
    for vals in valid_candidates:
        if time.time() > deadline:
            break
            
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
            
        sp = {}
        for u, v in combinations(range(n), 2):
            a = vals[u]
            b = vals[v]
            sp[(u, v)] = (a + b, a * b)
            
        score, edges = _local_search(vals, sp, edge_list, m, max_iter=5000, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_result = (vals, edges)
            if best_score <= 2:
                break
                
    if best_result is None:
        vals = valid_candidates[0] if valid_candidates else list(range(1, n + 1))
        edges = edge_list[:m]
        values = vals
    else:
        values, edges = best_result
        
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert min(values) >= 1 and max(values) <= max_val
    assert len(edges) == m
    assert all(u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': values, 'edges': edges}
