import time
import random
import math
from itertools import combinations

# Deadline guard: 90s limit, use 85s for safety
START_TIME = time.time()
DEADLINE = START_TIME + 85.0

def _check_time():
    return time.time() > DEADLINE

def _score(values, edges):
    """Calculate the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _greedy_edge_selection(values, n_edges=35):
    """
    Deterministic greedy: iteratively pick the edge that minimizes
    the increase in max(distinct_sums, distinct_products).
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products
    edge_sum = {}
    edge_prod = {}
    for u, v in all_edges:
        edge_sum[(u, v)] = values[u] + values[v]
        edge_prod[(u, v)] = values[u] * values[v]
    
    selected = set()
    s_counts = {}
    p_counts = {}
    
    for _ in range(n_edges):
        best_edge = None
        best_score = float('inf')
        
        for e in all_edges:
            if e in selected:
                continue
            s = edge_sum[e]
            p = edge_prod[e]
            
            # Marginal increase in distinct count
            ds = 0 if s in s_counts else 1
            dp = 0 if p in p_counts else 1
            
            new_s_count = len(s_counts) + ds
            new_p_count = len(p_counts) + dp
            new_score = max(new_s_count, new_p_count)
            
            # Tie-break: prefer edges whose sum/prod are already common
            s_freq = s_counts.get(s, 0)
            p_freq = p_counts.get(p, 0)
            
            if (new_score, -s_freq, -p_freq) < (best_score, 0, 0) or best_edge is None:
                if best_edge is None or new_score < best_score:
                    best_score = new_score
                    best_edge = e
                elif new_score == best_score:
                    # Prefer higher frequency
                    if (s_freq + p_freq) > (edge_sum.get(best_edge, 0) and 0):
                        # Simplified tie-break
                        if s_freq + p_freq > 0:
                            best_edge = e
        
        if best_edge is None:
            break
            
        selected.add(best_edge)
        s = edge_sum[best_edge]
        p = edge_prod[best_edge]
        s_counts[s] = s_counts.get(s, 0) + 1
        p_counts[p] = p_counts.get(p, 0) + 1
    
    return list(selected), max(len(s_counts), len(p_counts))

def _local_search_edges(values, initial_edges, n_edges=35, max_iters=5000):
    """
    Local search: swap edges to reduce max(distinct_sums, distinct_products).
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    edge_sum = {}
    edge_prod = {}
    for u, v in all_edges:
        edge_sum[(u, v)] = values[u] + values[v]
        edge_prod[(u, v)] = values[u] * values[v]
    
    current_edges = set(initial_edges)
    s_counts = {}
    p_counts = {}
    for e in current_edges:
        s = edge_sum[e]
        p = edge_prod[e]
        s_counts[s] = s_counts.get(s, 0) + 1
        p_counts[p] = p_counts.get(p, 0) + 1
    
    current_score = max(len(s_counts), len(p_counts))
    best_edges = set(current_edges)
    best_score = current_score
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    no_improve = 0
    
    for it in range(max_iters):
        if _check_time():
            break
        if T < min_T:
            break
            
        # Remove a random edge
        removed = random.choice(list(current_edges))
        s_rm = edge_sum[removed]
        p_rm = edge_prod[removed]
        
        s_counts[s_rm] -= 1
        if s_counts[s_rm] == 0:
            del s_counts[s_rm]
        p_counts[p_rm] -= 1
        if p_counts[p_rm] == 0:
            del p_counts[p_rm]
        
        base_s = len(s_counts)
        base_p = len(p_counts)
        
        # Try to add a different edge
        cand = None
        for _ in range(20):
            cand = random.choice(all_edges)
            if cand not in current_edges:
                break
        if cand is None or cand in current_edges:
            # Restore
            s_counts[s_rm] = s_counts.get(s_rm, 0) + 1
            p_counts[p_rm] = p_counts.get(p_rm, 0) + 1
            continue
            
        s_add = edge_sum[cand]
        p_add = edge_prod[cand]
        
        new_s = base_s + (0 if s_add in s_counts else 1)
        new_p = base_p + (0 if p_add in p_counts else 1)
        new_score = max(new_s, new_p)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_edges.discard(removed)
            current_edges.add(cand)
            s_counts[s_add] = s_counts.get(s_add, 0) + 1
            p_counts[p_add] = p_counts.get(p_add, 0) + 1
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                no_improve = 0
            else:
                no_improve += 1
        else:
            s_counts[s_rm] = s_counts.get(s_rm, 0) + 1
            p_counts[p_rm] = p_counts.get(p_rm, 0) + 1
        
        T *= cooling
    
    return best_score, list(best_edges)

def _generate_candidates():
    """
    Generate value sets with strong multiplicative and additive structure.
    Focus on numbers with many divisors and small prime factorizations.
    """
    candidates = []
    n = 14
    
    # Highly structured: powers of 2 and 3
    grid_23 = []
    for a in range(0, 6):
        for b in range(0, 6):
            val = (2 ** a) * (3 ** b)
            if val <= 10000 and val >= 1:
                grid_23.append(val)
    grid_23 = sorted(set(grid_23))
    
    if len(grid_23) >= 14:
        candidates.append(grid_23[:14])
    
    # Add 5 as a third prime
    grid_235 = []
    for a in range(0, 4):
        for b in range(0, 4):
            for c in range(0, 3):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 10000 and val >= 1:
                    grid_235.append(val)
    grid_235 = sorted(set(grid_235))
    
    if len(grid_235) >= 14:
        candidates.append(grid_235[:14])
    
    # Curated sets with known good properties
    curated = [
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 48],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32, 36, 48, 72],
        [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 27, 32, 36, 54],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 24, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 30],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 30, 36, 48, 60],
        [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 30, 36, 48, 72],
        [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48, 60, 72],
        [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48, 72, 96],
    ]
    candidates.extend(curated)
    
    # Random samples from structured grids
    random.seed(42)
    for _ in range(30):
        if len(grid_23) >= 14:
            vals = sorted(random.sample(grid_23, 14))
            candidates.append(vals)
    for _ in range(30):
        if len(grid_235) >= 14:
            vals = sorted(random.sample(grid_235, 14))
            candidates.append(vals)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        if len(c) != 14:
            continue
        if len(set(c)) != 14:
            continue
        if min(c) < 1 or max(c) > 10000:
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(list(c))
    
    return unique_candidates

def construct():
    n = 14
    m = 35
    
    candidates = _generate_candidates()
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Phase 1: Greedy + local search on each candidate
    for vals in candidates:
        if _check_time():
            break
            
        # Quick greedy to get initial solution
        greedy_edges, greedy_score = _greedy_edge_selection(vals, n_edges=m)
        
        # Local search to improve
        ls_score, ls_edges = _local_search_edges(vals, greedy_edges, n_edges=m, max_iters=3000)
        
        if ls_score < best_score:
            best_score = ls_score
            best_values = vals
            best_edges = ls_edges
            
            if best_score <= 5:
                break
    
    # Phase 2: If time remains, try value perturbation on best
    if best_values is not None and not _check_time():
        current_vals = best_values[:]
        current_edges = best_edges
        current_score = best_score
        
        for _ in range(50):
            if _check_time():
                break
            idx = random.randint(0, n - 1)
            old_val = current_vals[idx]
            new_val = random.randint(1, 60)
            if new_val == old_val or new_val in current_vals:
                continue
                
            new_vals = current_vals[:]
            new_vals[idx] = new_val
            
            # Quick check with current edges
            new_score = _score(new_vals, current_edges)
            
            if new_score < current_score:
                current_vals = new_vals
                current_score = new_score
                # Re-optimize edges
                g_edges, g_score = _greedy_edge_selection(current_vals, n_edges=m)
                o_score, o_edges = _local_search_edges(current_vals, g_edges, n_edges=m, max_iters=2000)
                if o_score < current_score:
                    current_score = o_score
                    current_edges = o_edges
                    best_score = current_score
                    best_values = current_vals
                    best_edges = current_edges

    # Final validation
    if best_values is None:
        best_values = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
        best_edges, _ = _greedy_edge_selection(best_values, n_edges=m)
        
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Sanity checks
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert min(values) >= 1 and max(values) <= 10000
    assert len(edges) == m
    assert all(u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': values, 'edges': edges}
