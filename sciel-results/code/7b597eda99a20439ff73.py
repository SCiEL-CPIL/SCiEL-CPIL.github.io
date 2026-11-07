"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _evaluate(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _greedy_edge_select(values, m):
    """Greedy: pick m edges minimizing marginal increase in distinct sums/products."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    sums_set = set()
    prods_set = set()
    selected = []
    
    # Precompute sum and product for each edge
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        i, j = e
        edge_sums[e] = values[i] + values[j]
        edge_prods[e] = values[i] * values[j]
    
    # Greedy: at each step, pick the edge that minimizes the increase in max(|sums|, |prods|)
    # Use a simpler metric: marginal increase in |sums| + |prods|
    remaining = list(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_cost = float('inf')
        best_new_sums = None
        best_new_prods = None
        
        for e in remaining:
            s = edge_sums[e]
            p = edge_prods[e]
            new_sums = len(sums_set | {s})
            new_prods = len(prods_set | {p})
            cost = max(new_sums, new_prods)
            if cost < best_cost:
                best_cost = cost
                best_edge = e
                best_new_sums = s
                best_new_prods = p
        
        if best_edge is None:
            break
        
        sums_set.add(best_new_sums)
        prods_set.add(best_new_prods)
        selected.append(best_edge)
        remaining.remove(best_edge)
    
    return selected

def _try_value_set(values, m):
    """Try a specific value set with greedy edge selection."""
    edges = _greedy_edge_select(values, m)
    if len(edges) < m:
        return None, None
    score = _evaluate(values, edges)
    return score, edges

def construct():
    start_time = time.time()
    n = 14
    m = 35
    
    # Candidate value sets designed for collisions
    candidates = []
    
    # Set 1: Powers of 2 (baseline - no collisions, bad)
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # Set 2: Small integers 1-14 (some additive collisions)
    candidates.append(list(range(1, 15)))
    
    # Set 3: Multiples with structure
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # Set 4: More multiplicative structure
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Set 5: Geometric-ish with collisions
    candidates.append([1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48])
    
    # Set 6: Densely packed small numbers
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Set 7: Multiples of small numbers
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36])
    
    # Set 8: Mix for both additive and multiplicative collisions
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27])
    
    # Set 9: Highly composite numbers
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48])
    
    # Set 10: Arithmetic progression with gcd structure
    candidates.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    
    # Set 11: Binary-like but with collisions
    candidates.append([1, 2, 3, 5, 6, 10, 11, 15, 16, 21, 22, 26, 27, 31])
    
    # Set 12: Optimized for product collisions (many factor pairs)
    candidates.append([1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 27, 32, 36, 48])
    
    best_score = float('inf')
    best_result = None
    
    for values in candidates:
        if time.time() - start_time > DEADLINE:
            break
        
        # Validate: 14 distinct positive integers <= 10000
        if len(values) != n:
            continue
        if len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        
        score, edges = _try_value_set(values, m)
        if score is not None and score < best_score:
            best_score = score
            best_result = {
                'values': list(values),
                'edges': [list(e) for e in edges]
            }
    
    # If no result found, use fallback
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
