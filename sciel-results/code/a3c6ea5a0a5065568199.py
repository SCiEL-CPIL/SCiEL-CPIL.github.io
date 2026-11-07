"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(e):
    a, b = e
    return (min(a, b), max(a, b))

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_biclique(remaining_edges, adj, n, deadline):
    """Find a large biclique within remaining_edges using common-neighbor heuristic."""
    if not remaining_edges:
        return None
    
    # Build edge set for fast lookup
    edge_set = set(_edge_key(e) for e in remaining_edges)
    
    # For each vertex, find its neighbors in the remaining edge subgraph
    # A vertex's "degree" in remaining edges
    deg = {v: 0 for v in range(n)}
    for u, v in remaining_edges:
        deg[u] += 1
        deg[v] += 1
    
    # Try to find biclique starting from high-degree vertices
    best_biclique = None
    best_size = 0
    
    # Get vertices sorted by degree in remaining subgraph
    verts_sorted = sorted(range(n), key=lambda x: -deg[x])
    
    # Limit search to top candidates for time efficiency
    top_k = min(15, len(verts_sorted))
    
    for i in range(top_k):
        if time.time() > deadline:
            break
        u = verts_sorted[i]
        if deg[u] < 2:
            continue
        
        # Neighbors of u in remaining edges
        neighbors_of_u = [v for v in range(n) if deg[v] > 0 and _edge_key((u, v)) in edge_set]
        
        # Try to build biclique with u on left side
        # Left side starts with {u}, right side = neighbors of u
        # Then expand: for each vertex w not in left, check if w connects to ALL of right
        left = {u}
        right = set(neighbors_of_u)
        
        # Greedily expand left side
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in left or w in right:
                    continue
                # Check if w connects to all vertices in right
                all_connected = all(_edge_key((w, r)) in edge_set for r in right)
                if all_connected and right:
                    left.add(w)
                    changed = True
        
        # Also try expanding right side
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in left or w in right:
                    continue
                all_connected = all(_edge_key((l, w)) in edge_set for l in left)
                if all_connected and left:
                    right.add(w)
                    changed = True
        
        size = len(left) * len(right)
        if size > best_size:
            best_size = size
            best_biclique = (left, right)
    
    if best_biclique is None:
        # Fallback: single edge
        u, v = remaining_edges[0]
        return ({u}, {v})
    
    left, right = best_biclique
    return (left, right)

def _extract_edges_from_biclique(biclique, edge_set):
    """Get the set of edges in edge_set that are cross-edges of the biclique."""
    left, right = biclique
    edges = set()
    for u in left:
        for v in right:
            k = _edge_key((u, v))
            if k in edge_set:
                edges.add(k)
    return edges

def _validate_partition(bicliques, all_edges):
    """Check that bicliques form a valid partition of all edges."""
    all_edge_set = set(_edge_key(e) for e in all_edges)
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        # Check disjointness of sides within this biclique
        if left & right:
            return False
        for u in left:
            for v in right:
                k = _edge_key((u, v))
                if k not in all_edge_set:
                    return False
                if k in covered:
                    return False
                covered.add(k)
    return covered == all_edge_set

def _try_merge(bicliques, all_edges, deadline):
    """Try to merge pairs of bicliques if compatible."""
    merged = True
    while merged and time.time() < deadline:
        merged = False
        bicliques = list(bicliques)
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try all 4 combinations of which side goes where
                for combo in [(li | lj, ri | rj), (li | rj, ri | lj), (ri | lj, li | rj), (ri | rj, li | lj)]:
                    new_left, new_right = combo
                    if not new_left or not new_right:
                        continue
                    if new_left & new_right:
                        continue
                    
                    # Check if this merged biclique is valid: all cross pairs must be edges
                    valid = True
                    for u in new_left:
                        for v in new_right:
                            k = _edge_key((u, v))
                            if k not in all_edges:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check that the merged biclique's cross-edges are exactly the union
                    # of the two original bicliques' cross-edges
                    cross_merged = set()
                    for u in new_left:
                        for v in new_right:
                            cross_merged.add(_edge_key((u, v)))
                    
                    cross_i = set()
                    for u in li:
                        for v in ri:
                            cross_i.add(_edge_key((u, v)))
                    cross_j = set()
                    for u in lj:
                        for v in rj:
                            cross_j.add(_edge_key((u, v)))
                    
                    if cross_merged == cross_i | cross_j:
                        # Valid merge
                        bicliques[i] = (new_left, new_right)
                        bicliques.pop(j)
                        merged = True
                        break
            if merged:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edges = set(_edge_key(e) for e in edges)
    adj = _build_adj(n, edges)
    
    # Greedy extraction
    remaining = set(all_edges)
    bicliques = []
    
    while remaining and time.time() < deadline:
        # Build list of remaining edges
        rem_list = [(u, v) for u, v in remaining]
        
        biclique = _find_max_biclique(rem_list, adj, n, deadline)
        if biclique is None:
            break
        
        left, right = biclique
        cross_edges = _extract_edges_from_biclique(biclique, remaining)
        
        if not cross_edges:
            # Shouldn't happen, but safety
            break
        
        bicliques.append((left, right))
        remaining -= cross_edges
    
    # If we still have remaining edges (shouldn't happen), add them as single-edge bicliques
    for u, v in remaining:
        bicliques.append(({u}, {v}))
    
    # Try to merge
    bicliques = _try_merge(bicliques, all_edges, deadline)
    
    # Validate
    if not _validate_partition(bicliques, edges):
        # Fallback to seed solution
        return [[[u],[v for a,v in edges if a==u]] for u in range(n) if any(a==u for a,v in edges)]
    
    # Convert to proper format
    result = []
    for left, right in bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
