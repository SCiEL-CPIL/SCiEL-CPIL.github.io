"""Rewritten program: bitmask-based exhaustive search with local search polish."""
from itertools import combinations
import time

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adjacency_bitmasks():
    n = P['n']
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _distinct_degree_count_mask(mask, adj_mask):
    """Count distinct degrees in induced subgraph represented by bitmask."""
    degrees = []
    for v in range(P['n']):
        if mask & (1 << v):
            deg = bin(mask & adj_mask[v]).count('1')
            degrees.append(deg)
    return len(set(degrees))

def _local_search(mask, adj_mask, deadline):
    """Local search: try swapping vertices to improve distinct degree count."""
    n = P['n']
    min_v = P['minimum_vertices']
    best_mask = mask
    best_score = _distinct_degree_count_mask(mask, adj_mask)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Try all single swaps
        for out_v in range(n):
            if not (mask & (1 << out_v)):
                continue
            for in_v in range(n):
                if mask & (1 << in_v):
                    continue
                if time.time() > deadline:
                    return best_mask, best_score
                new_mask = mask | (1 << in_v) & ~(1 << out_v)
                # Check size constraint
                if bin(new_mask).count('1') < min_v:
                    continue
                score = _distinct_degree_count_mask(new_mask, adj_mask)
                if score > best_score:
                    best_score = score
                    best_mask = new_mask
                    improved = True
                    mask = new_mask
                    break
            if improved:
                break
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    n = P['n']
    min_v = P['minimum_vertices']
    adj_mask = _build_adjacency_bitmasks()
    
    all_vertices = list(range(n))
    best_score = 0
    best_mask = 0
    
    # Phase 1: Exhaustive search over subset sizes from large to small
    for size in range(n, min_v - 1, -1):
        if time.time() > deadline:
            break
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            score = _distinct_degree_count_mask(mask, adj_mask)
            if score > best_score:
                best_score = score
                best_mask = mask
                # Theoretical max for this size is min(size, n-1) but practically size
                if best_score == size:
                    # Found perfect: all vertices have distinct degrees
                    break
        if time.time() > deadline:
            break
        # If we found the max possible for any size >= min_v, we can stop
        if best_score >= size:
            break
    
    # Phase 2: Local search polish on the best solution found
    if best_mask > 0 and time.time() < deadline:
        polished_mask, polished_score = _local_search(best_mask, adj_mask, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_mask = polished_mask
    
    # Phase 3: If we haven't found high score, try random restarts with local search
    if best_score < 10 and time.time() < deadline - 5:
        import random
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            # Random subset of size between min_v and n
            sz = random.randint(min_v, n)
            subset = random.sample(all_vertices, sz)
            mask = 0
            for v in subset:
                mask |= (1 << v)
            score = _distinct_degree_count_mask(mask, adj_mask)
            if score > best_score:
                best_score = score
                best_mask = mask
            # Local search from this starting point
            if time.time() < deadline - 5:
                pm, ps = _local_search(mask, adj_mask, deadline)
                if ps > best_score:
                    best_score = ps
                    best_mask = pm
    
    # Convert best_mask to list of vertex indices
    result = [int(v) for v in range(n) if best_mask & (1 << v)]
    return result
# EVOLVE-BLOCK-END
