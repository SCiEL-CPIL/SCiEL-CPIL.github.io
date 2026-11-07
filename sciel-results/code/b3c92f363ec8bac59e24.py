"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degree_count(subset, adj):
    """Return the number of distinct degrees in the induced subgraph on `subset`."""
    s = set(subset)
    degrees = []
    for v in s:
        deg = sum(1 for u in s if u in adj[v] and u != v)
        degrees.append(deg)
    return len(set(degrees))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    adj = _build_adj()
    n = P['n']
    
    best_score = -1
    best_subset = list(range(8))
    
    # Brute force ALL subsets of size 8 through 17
    # Total subsets: C(17,8)+...+C(17,17) ~ 78,000
    # Each evaluation is fast (O(k^2) with k<=17)
    
    for size in range(8, n + 1):
        for subset in itertools.combinations(range(n), size):
            s = set(subset)
            degrees = []
            for v in s:
                deg = 0
                for u in s:
                    if u != v and u in adj[v]:
                        deg += 1
                degrees.append(deg)
            score = len(set(degrees))
            if score > best_score:
                best_score = score
                best_subset = list(subset)
        
        # Time check
        if time.time() > deadline:
            break
    
    # Convert to proper types
    result = [int(v) for v in best_subset]
    return result

# EVOLVE-BLOCK-END
