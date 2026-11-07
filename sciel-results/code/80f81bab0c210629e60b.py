"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _count_distinct_degrees(subset, adj):
    """Count distinct degrees in the induced subgraph on subset."""
    subset_set = set(subset)
    degrees = set()
    for v in subset:
        deg = sum(1 for u in subset if u != v and u in adj[v])
        degrees.add(deg)
    return len(degrees)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    n = P['n']
    min_v = P['minimum_vertices']
    adj = _build_adj()
    
    # Precompute adjacency as bitmasks for fast degree counting
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_vertices = list(range(n))
    
    best_score = 0
    best_subset = list(range(min_v))  # fallback
    
    # Start with the seed answer
    seed_subset = list(range(9))
    best_score = _count_distinct_degrees(seed_subset, adj)
    best_subset = seed_subset
    
    # Try all subsets of size 8 to 17, prioritizing smaller subsets first
    # since they're faster to evaluate and may have more spread
    for size in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            # Quick check: count distinct degrees
            subset_set = set(subset)
            degs = set()
            for v in subset:
                # Count neighbors of v within subset using bitmask
                mask = adj_mask[v] & 0
                # Build subset mask
                smask = 0
                for u in subset:
                    smask |= (1 << u)
                # Remove v from smask
                smask_no_v = smask & ~(1 << v)
                deg = bin(adj_mask[v] & smask_no_v).count('1')
                degs.add(deg)
                if len(degs) + (size - list(degs).__len__()) <= best_score:
                    # Can't beat best, but this check is expensive; skip for now
                    pass
            score = len(degs)
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                if best_score >= size:  # Can't do better than size distinct degrees
                    # Actually max distinct degrees is size (one per vertex)
                    # But degrees range 0..size-1, so max is size
                    if best_score == size:
                        # Perfect: all degrees distinct
                        return [int(x) for x in best_subset]
    
    # If we found a good answer, do a local search to try to improve
    if time.time() < deadline - 5:
        # Local search: try swapping vertices in/out
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current = list(best_subset)
            current_set = set(current)
            for v in current:
                if time.time() > deadline - 5:
                    break
                for w in all_vertices:
                    if w in current_set:
                        continue
                    # Try swapping v for w
                    new_subset = [x for x in current if x != v] + [w]
                    new_set = set(new_subset)
                    degs = set()
                    for u in new_subset:
                        smask = 0
                        for x in new_subset:
                            smask |= (1 << x)
                        smask_no_u = smask & ~(1 << u)
                        deg = bin(adj_mask[u] & smask_no_u).count('1')
                        degs.add(deg)
                    if len(degs) > best_score:
                        best_score = len(degs)
                        best_subset = new_subset
                        current = new_subset
                        current_set = new_set
                        improved = True
                        break
    
    return [int(x) for x in best_subset]
# EVOLVE-BLOCK-END
