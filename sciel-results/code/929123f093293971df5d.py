"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _expand_biclique(L, R, adj, remaining):
    """Expand biclique (L,R) by adding vertices that have all required edges."""
    L = set(L)
    R = set(R)
    changed = True
    while changed:
        changed = False
        # Add to L: vertex x not in L∪R that connects to all of R
        for x in range(P['n']):
            if x not in L and x not in R:
                if all(adj[x].superset_check(y) for y in R):
                    L.add(x)
                    changed = True
        # Add to R: vertex x not in L∪R that connects to all of L
        for x in range(P['n']):
            if x not in L and x not in R:
                if all(adj[x].superset_check(y) for y in L):
                    R.add(x)
                    changed = True
    return L, R

def _find_best_biclique(remaining, adj, rng=None):
    """Find the largest biclique in the remaining edge set."""
    if not remaining:
        return None
    
    best = None
    best_edges = 0
    
    edge_list = list(remaining)
    if rng is not None:
        rng.shuffle(edge_list)
    
    # Try a subset of seeds for speed
    max_seeds = min(len(edge_list), 30)
    for i in range(max_seeds):
        u, v = edge_list[i]
        L, R = _expand_biclique({u}, {v}, adj, remaining)
        
        # Verify all cross edges are in remaining
        edge_count = 0
        valid = True
        for a in L:
            for b in R:
                if (a, b) in remaining or (b, a) in remaining:
                    edge_count += 1
                else:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and edge_count > best_edges:
            best_edges = edge_count
            best = (L, R)
    
    return best

def _greedy_partition(remaining, adj, deadline, rng=None):
    """Greedy: repeatedly extract largest biclique."""
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        result = _find_best_biclique(remaining, adj, rng)
        
        if result is None:
            # Fallback: take one edge
            u, v = next(iter(remaining))
            bicliques.append((frozenset([u]), frozenset([v])))
            remaining.discard((u, v))
            remaining.discard((v, u))
            continue
        
        L, R = result
        bicliques.append((frozenset(L), frozenset(R)))
        
        # Remove all cross edges
        to_remove = set()
        for a in L:
            for b in R:
                to_remove.add((a, b))
                to_remove.add((b, a))
        remaining -= to_remove
    
    return bicliques

def _validate(bicliques, edges_set):
    """Check partition validity."""
    covered = set()
    for L, R in bicliques:
        if not L or not R:
            return False
        if L & R:
            return False
        for u in L:
            for v in R:
                if (u, v) not in edges_set and (v, u) not in edges_set:
                    return False
                covered.add(frozenset([u, v]))
    return covered == edges_set

def _to_result(bicliques):
    """Convert to required output format."""
    result = []
    for L, R in bicliques:
        left = [int(v) for v in sorted(L)]
        right = [int(v) for v in sorted(R)]
        result.append([left, right])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 78  # 78s safety margin (90s limit)
    
    n = P['n']
    edges_set = set()
    for u, v in P['edges']:
        edges_set.add((u, v))
        edges_set.add((v, u))
    
    adj = _build_adj()
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multiple greedy runs with different random orderings
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        remaining = set(edges_set)
        bicliques = _greedy_partition(remaining, adj, deadline, rng)
        if bicliques and _validate(bicliques, edges_set):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques
    
    # Fallback: per-vertex stars
    if best_bicliques is None:
        bicliques = []
        for u in range(n):
            neighbors = list(adj[u])
            if neighbors:
                bicliques.append((frozenset([u]), frozenset(neighbors)))
        best_bicliques = bicliques
    
    return _to_result(best_bicliques)
# EVOLVE-BLOCK-END
