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

def _find_max_biclique(remaining_edges, adj):
    """Find a large biclique from remaining edges using seed-based expansion."""
    if not remaining_edges:
        return None
    
    best_L = None
    best_R = None
    best_size = 0
    
    # Sample edges to seed the search
    edge_list = list(remaining_edges)
    if len(edge_list) > 20:
        random.shuffle(edge_list)
        edge_list = edge_list[:20]
    
    for u, v in edge_list:
        L = {u}
        R = {v}
        
        # Expand greedily
        changed = True
        while changed:
            changed = False
            # Try to add to L: x must be connected to all in R
            # and x not in R (disjoint sides)
            for x in range(P['n']):
                if x not in L and x not in R:
                    # Check if x is connected to all in R via remaining edges?
                    # No, the biclique definition requires edges to exist in the ORIGINAL graph.
                    # But we are partitioning REMAINING edges.
                    # The condition is: for all v in R, (x,v) must be in remaining_edges.
                    # If we pick a biclique, we remove those edges.
                    # So we must ensure (x,v) is in remaining_edges for all v in R.
                    is_connected = True
                    for y in R:
                        if (x, y) not in remaining_edges and (y, x) not in remaining_edges:
                            is_connected = False
                            break
                    if is_connected:
                        L.add(x)
                        changed = True
            
            # Try to add to R: x must be connected to all in L
            for x in range(P['n']):
                if x not in L and x not in R:
                    is_connected = True
                    for y in L:
                        if (x, y) not in remaining_edges and (y, x) not in remaining_edges:
                            is_connected = False
                            break
                    if is_connected:
                        R.add(x)
                        changed = True
        
        size = len(L) * len(R)
        if size > best_size:
            best_size = size
            best_L = set(L)
            best_R = set(R)
            
    if best_L is None or not best_L or not best_R:
        return None
        
    return best_L, best_R

def _greedy_partition(remaining_edges, adj, deadline, seed=0):
    """Greedy max-biclique extraction."""
    rng = random.Random(seed)
    bicliques = []
    
    while remaining_edges:
        if time.time() > deadline:
            break
        
        # Find max biclique
        result = _find_max_biclique(remaining_edges, adj)
        if result is None:
            # Fallback: single edge
            u, v = next(iter(remaining_edges))
            bicliques.append((frozenset([u]), frozenset([v])))
            remaining_edges.discard((u, v))
            remaining_edges.discard((v, u))
            continue
        
        L, R = result
        bicliques.append((frozenset(L), frozenset(R)))
        
        # Remove all cross edges
        to_remove = set()
        for u in L:
            for v in R:
                to_remove.add((u, v))
                to_remove.add((v, u))
        
        # Ensure we actually remove something
        if not (to_remove & remaining_edges):
             # This shouldn't happen if _find_max_biclique is correct,
             # but if it does, force a single edge removal to avoid infinite loop
             u, v = next(iter(remaining_edges))
             remaining_edges.discard((u, v))
             remaining_edges.discard((v, u))
             continue

        remaining_edges -= to_remove
    
    return bicliques

def _validate(bicliques, edges_set):
    """Validate that bicliques form a valid partition."""
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
    """Convert internal representation to required format."""
    result = []
    for L, R in bicliques:
        left = [int(v) for v in sorted(L)]
        right = [int(v) for v in sorted(R)]
        result.append([left, right])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = P['n']
    edges_set = set()
    for u, v in P['edges']:
        edges_set.add((u, v))
        edges_set.add((v, u))
    
    adj = _build_adj()
    
    # Try multiple strategies and seeds
    best_bicliques = None
    best_count = float('inf')
    
    # Strategy 1: Greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        remaining = set(edges_set)
        bicliques = _greedy_partition(remaining, adj, deadline, seed=seed)
        if not bicliques:
            continue
        if _validate(bicliques, edges_set):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques
    
    # Strategy 2: Deterministic greedy (no randomness in seed selection)
    if time.time() < deadline:
        remaining = set(edges_set)
        bicliques = _greedy_partition(remaining, adj, deadline, seed=0)
        if bicliques and _validate(bicliques, edges_set):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques
    
    # Fallback: per-vertex stars
    if best_bicliques is None:
        bicliques = []
        remaining = set(edges_set)
        # Simple greedy edge removal for fallback
        while remaining:
            u, v = next(iter(remaining))
            bicliques.append((frozenset([u]), frozenset([v])))
            remaining.discard((u, v))
            remaining.discard((v, u))
        best_bicliques = bicliques
    
    return _to_result(best_bicliques)
# EVOLVE-BLOCK-END
