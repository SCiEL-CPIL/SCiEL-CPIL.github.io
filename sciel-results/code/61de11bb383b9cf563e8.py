"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _greedy_edges(values, max_edges=35):
    """Greedy: pick edges that minimize marginal increase in max(distinct_sums, distinct_products)."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    selected = []
    sums = set()
    prods = set()
    
    for e in all_edges:
        if len(selected) >= max_edges:
            break
        i, j = e
        s = values[i] + values[j]
        p = values[i] * values[j]
        # Check if adding this edge increases the max
        new_sums = len(sums | {s})
        new_prods = len(prods | {p})
        # Prefer edges that don't increase the count
        if s in sums and p in prods:
            selected.append(list(e))
            sums.add(s)
            prods.add(p)
    
    # If we don't have enough, add remaining
    if len(selected) < max_edges:
        for e in all_edges:
            if len(selected) >= max_edges:
                break
            if e not in selected:
                i, j = e
                s = values[i] + values[j]
                p = values[i] * values[j]
                selected.append(list(e))
                sums.add(s)
                prods.add(p)
    
    return selected

def _local_search(values, edges, max_iter=2000):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    edge_set = set(tuple(sorted(e)) for e in edges)
    current = _score(values, edges)
    
    for _ in range(max_iter):
        in_list = list(edge_set)
        out_list = [e for e in all_edges if e not in edge_set]
        if not out_list:
            break
        
        # Try a random swap
        old = random.choice(in_list)
        new = random.choice(out_list)
        
        edge_set.discard(old)
        edge_set.add(new)
        new_edges = [list(e) for e in edge_set]
        new_score = _score(values, new_edges)
        
        if new_score < current:
            current = new_score
        else:
            # Revert
            edge_set.discard(new)
            edge_set.add(old)
    
    return [list(e) for e in edge_set], current

def construct():
    start = time.time()
    deadline = start + 75
    
    best = None
    best_score = float('inf')
    
    candidates = [
        list(range(1, 15)),
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ]
    
    for vals in candidates:
        if time.time() > deadline:
            break
        for seed in [1, 42, 99]:
            if time.time() > deadline:
                break
            random.seed(seed)
            edges = _greedy_edges(vals)
            edges, s = _local_search(vals, edges, max_iter=1500)
            if s < best_score:
                best_score = s
                best = (vals, edges)
    
    if best is None:
        vals = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {'values': vals, 'edges': edges}
    
    return {'values': best[0], 'edges': best[1]}
# EVOLVE-BLOCK-END
