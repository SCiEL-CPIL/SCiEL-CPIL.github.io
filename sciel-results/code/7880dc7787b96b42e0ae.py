"""Rewritten program: exhaustive search over subsets to maximize distinct induced degrees."""
from itertools import combinations
import time

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _distinct_degree_count(subset, adj):
    """Count distinct degrees in the induced subgraph on subset."""
    subset_set = set(subset)
    degrees = []
    for v in subset:
        deg = sum(1 for u in subset_set if u in adj[v])
        degrees.append(deg)
    return len(set(degrees))

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    n = P['n']
    min_v = P['minimum_vertices']
    adj = _build_adjacency()
    
    all_vertices = list(range(n))
    
    best_score = 0
    best_subset = list(range(min_v))
    
    # Try all subset sizes from large to small (larger subsets can have more distinct degrees)
    # With 8 vertices, max distinct degrees is 8 (degrees 0..7)
    # With 9 vertices, max distinct degrees is 9 (degrees 0..8)
    # But degrees are bounded by subset_size - 1
    
    # Enumerate by size, starting from larger subsets
    for size in range(n, min_v - 1, -1):
        if time.time() > deadline:
            break
        count = 0
        for subset in combinations(all_vertices, size):
            count += 1
            if time.time() > deadline:
                break
            score = _distinct_degree_count(subset, adj)
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                # If we achieved the theoretical max for this size, we can stop
                if best_score == size:
                    return best_subset
            # Early termination: if we've found max possible for any size
            if best_score >= size:
                break
    
    # Ensure we return a valid list of Python ints
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
