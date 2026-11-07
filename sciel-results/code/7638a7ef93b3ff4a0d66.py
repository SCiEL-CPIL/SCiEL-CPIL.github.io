"""Rewritten program: efficient search to maximize distinct induced degrees."""
from itertools import combinations
import time

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adjacency():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _distinct_degree_count_mask(mask, adj):
    n = P['n']
    degrees = set()
    for v in range(n):
        if mask & (1 << v):
            deg = bin(mask & adj[v]).count('1')
            degrees.add(deg)
    return len(degrees)

def _local_search(mask, adj, deadline):
    """Improve mask via swaps and add/remove operations."""
    n = P['n']
    min_v = P['minimum_vertices']
    best_mask = mask
    best_score = _distinct_degree_count_mask(mask, adj)
    
    for it in range(200):
        if time.time() > deadline:
            break
        improved = False
        # Try swaps
        for v_in in range(n):
            if best_mask & (1 << v_in):
                continue
            for v_out in range(n):
                if not (best_mask & (1 << v_out)):
                    continue
                new_mask = (best_mask | (1 << v_in)) & ~(1 << v_out)
                score = _distinct_degree_count_mask(new_mask, adj)
                if score > best_score:
                    best_score = score
                    best_mask = new_mask
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Try adds
        for v in range(n):
            if best_mask & (1 << v):
                continue
            new_mask = best_mask | (1 << v)
            score = _distinct_degree_count_mask(new_mask, adj)
            if score > best_score:
                best_score = score
                best_mask = new_mask
                improved = True
                break
        if improved:
            continue
        # Try removes (only if size > min_v)
        if bin(best_mask).count('1') > min_v:
            for v in range(n):
                if not (best_mask & (1 << v)):
                    continue
                new_mask = best_mask & ~(1 << v)
                score = _distinct_degree_count_mask(new_mask, adj)
                if score > best_score:
                    best_score = score
                    best_mask = new_mask
                    improved = True
                    break
        if not improved:
            break
    
    return best_mask, best_score

def construct():
    deadline = time.time() + 85
    n = P['n']
    min_v = P['minimum_vertices']
    adj = _build_adjacency()
    
    best_score = 0
    best_mask = 0
    
    # Exhaustive search over all subsets of size >= min_v
    for size in range(n, min_v - 1, -1):
        if time.time() > deadline:
            break
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            score = _distinct_degree_count_mask(mask, adj)
            if score > best_score:
                best_score = score
                best_mask = mask
    
    # Local search polish
    if time.time() < deadline:
        best_mask, best_score = _local_search(best_mask, adj, deadline)
    
    result = [int(v) for v in range(n) if best_mask & (1 << v)]
    if len(result) < min_v:
        result = list(range(min_v))
    
    return result
# EVOLVE-BLOCK-END
