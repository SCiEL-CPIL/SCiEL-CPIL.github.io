"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

_DEADLINE = time.time() + 80

def _raw_score(values, edge_pairs):
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

def _precompute_sp(values, n):
    """Precompute (sum, product) for every edge."""
    sp = {}
    for u in range(n):
        for v in range(u + 1, n):
            a = values[u]
            b = values[v]
            sp[(u, v)] = (a + b, a * b)
    return sp

def _greedy_edge_select(sp, edge_list, m):
    """
    Greedy: at each step, pick the edge that minimizes the increase in
    max(|distinct_sums|, |distinct_prods|). Ties broken by preferring
    edges whose (sum, product) pair is most common overall.
    """
    # Precompute frequency of each sum and product
    sum_freq = defaultdict(int)
    prod_freq = defaultdict(int)
    pair_freq = defaultdict(int)
    for e in edge_list:
        s, p = sp[e]
        sum_freq[s] += 1
        prod_freq[p] += 1
        pair_freq[(s, p)] += 1
    
    selected = []
    sums = set()
    prods = set()
    remaining = list(edge_list)
    
    for _ in range(m):
        best_edge = None
        best_metric = (999999, 999999, 0)  # (new_max, new_sums, -pair_freq)
        for e in remaining:
            s, p = sp[e]
            new_sums = len(sums) + (1 if s not in sums else 0)
            new_prods = len(prods) + (1 if p not in prods else 0)
            new_max = max(new_sums, new_prods)
            # Prefer lower new_max, then fewer total distinct, then higher pair frequency
            metric = (new_max, new_sums + new_prods, -pair_freq[(s, p)])
            if metric < best_metric:
                best_metric = metric
                best_edge = e
        
        if best_edge is None:
            break
        s, p = sp[best_edge]
        selected.append(best_edge)
        sums.add(s)
        prods.add(p)
        remaining.remove(best_edge)
    
    return selected, sums, prods

def _local_search_edges(sp, edge_list, m, max_iter, deadline, seed=42):
    """
    Local search: start from greedy, then do 1-swap moves.
    Remove one edge, add another, accept if score doesn't worsen.
    """
    rng = random.Random(seed)
    n = len(values) if 'values' in dir() else len(edge_list[0][0]) + 1  # fallback
    # Actually we need n from edge_list
    max_idx = 0
    for e in edge_list:
        max_idx = max(max_idx, e[0], e[1])
    n = max_idx + 1
    
    all_edges_set = set(edge_list)
    
    # Start with greedy
    edges, sums, prods = _greedy_edge_select(sp, edge_list, m)
    edge_set = set(edges)
    current_score = max(len(sums), len(prods))
    best_score = current_score
    best_edges = list(edges)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if current_score <= 3:
            break
        
        # Pick random edge to remove
        u, v = rng.choice(list(edge_set))
        s_old = sp[(u, v)][0]
        p_old = sp[(u, v)][1]
        
        # Compute current counts
        s_counts = defaultdict(int)
        p_counts = defaultdict(int)
        for (x, y) in edge_set:
            sx, px = sp[(x, y)]
            s_counts[sx] += 1
            p_counts[px] += 1
        
        # Remove
        s_counts[s_old] -= 1
        p_counts[p_old] -= 1
        if s_counts[s_old] == 0:
            del s_counts[s_old]
        if p_counts[p_old] == 0:
            del p_counts[p_old]
        
        # Try adding candidates
        candidates = [e for e in all_edges_set if e not in edge_set]
        if not candidates:
            break
        
        # Sample for speed
        sample_size = min(50, len(candidates))
        sample = rng.sample(candidates, sample_size)
        
        best_add = None
        best_new_score = current_score
        
        for cand in sample:
            sx, px = sp[cand]
            new_s = len(s_counts) + (1 if sx not in s_counts else 0)
            new_p = len(p_counts) + (1 if px not in p_counts else 0)
            new_score = max(new_s, new_p)
            if new_score < best_new_score:
                best_new_score = new_score
                best_add = cand
        
        if best_add is not None and best_new_score < current_score:
            edge_set.discard((u, v))
            edge_set.add(best_add)
            sx, px = sp[best_add]
            s_counts[sx] = s_counts.get(sx, 0) + 1
            p_counts[px] = p_counts.get(px, 0) + 1
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edge_set)
    
    return best_score, best_edges

def _generate_value_candidates():
    """Generate diverse value sets with rich additive/multiplicative structure."""
    n = 14
    candidates = []
    
    # Small integers with many relations
    base_sets = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 30],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18],
        [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 9, 10, 12, 14, 15, 16, 18, 20, 25],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ]
    
    for vals in base_sets:
        if len(set(vals)) == n and all(1 <= v <= 10000 for v in vals):
            candidates.append(list(vals))
    
    # Random small sets
    rng = random.Random(123)
    seen = set(tuple(sorted(c)) for c in candidates)
    for _ in range(50):
        vals = sorted(rng.sample(range(1, 50), n))
        key = tuple(vals)
        if key not in seen:
            seen.add(key)
            candidates.append(vals)
    
    # Sets with many shared factors (multiplicative collisions)
    for _ in range(30):
        # Pick a base and multiply by small factors
        base = rng.sample(range(1, 15), 7)
        vals = []
        for b in base:
            vals.append(b)
            vals.append(b * 2)
        # Deduplicate and ensure 14 distinct
        vals = sorted(set(vals))
        if len(vals) < 14:
            # Fill with random
            extra = rng.sample(range(1, 100), 14 - len(vals))
            vals = sorted(set(vals + extra))
        if len(vals) == 14 and all(1 <= v <= 10000 for v in vals):
            key = tuple(vals)
            if key not in seen:
                seen.add(key)
                candidates.append(vals)
    
    return candidates

def construct():
    n = 14
    m = 35
    deadline = time.time() + 78
    
    edge_list = _all_edges(n)
    candidates = _generate_value_candidates()
    
    best_score = m  # worst case
    best_result = None
    
    # Try each candidate value set
    for vals in candidates:
        if time.time() > deadline:
            break
        
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > 10000:
            continue
        
        sp = _precompute_sp(vals, n)
        
        # Run local search with a few different seeds
        for seed in [42, 123, 456]:
            if time.time() > deadline:
                break
            score, edges = _local_search_edges(sp, edge_list, m, max_iter=2000, deadline=deadline, seed=seed)
            if score < best_score:
                best_score = score
                best_result = (list(vals), list(edges))
                if best_score <= 3:
                    break
    
    if best_result is None:
        vals = candidates[0] if candidates else list(range(1, n + 1))
        edges = edge_list[:m]
    else:
        vals, edges = best_result
    
    # Ensure correct types
    values = [int(v) for v in vals]
    edges = [[int(u), int(v)] for u, v in edges]
    
    # Final validation
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert min(values) >= 1 and max(values) <= 10000
    assert len(edges) == m
    assert all(u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
