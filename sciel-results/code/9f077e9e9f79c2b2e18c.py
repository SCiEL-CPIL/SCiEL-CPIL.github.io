"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _grow_biclique(seed_u, seed_v, adj, uncovered_edges_set, n):
    """Grow a biclique starting from edge (seed_u, seed_v)."""
    L = {seed_u}
    R = {seed_v}
    changed = True
    while changed:
        changed = False
        # Try to add to L: vertex x not in L∪R, connected to all in R
        for x in range(n):
            if x not in L and x not in R:
                if R.issubset(adj[x]):
                    # Check that all edges (x, r) for r in R are uncovered
                    all_uncovered = all((min(x, r), max(x, r)) in uncovered_edges_set for r in R)
                    if all_uncovered:
                        L.add(x)
                        changed = True
        # Try to add to R: vertex y not in L∪R, connected to all in L
        for y in range(n):
            if y not in L and y not in R:
                if L.issubset(adj[y]):
                    all_uncovered = all((min(l, y), max(l, y)) in uncovered_edges_set for l in L)
                    if all_uncovered:
                        R.add(y)
                        changed = True
    # Only keep edges that are in uncovered_edges_set
    valid_L = set()
    for l in L:
        if all((min(l, r), max(l, r)) in uncovered_edges_set for r in R):
            valid_L.add(l)
    valid_R = set()
    for r in R:
        if all((min(l, r), max(l, r)) in uncovered_edges_set for l in valid_L):
            valid_R.add(r)
    if not valid_L or not valid_R:
        return None
    return (valid_L, valid_R)

def _biclique_edges(L, R):
    """Get all edges between L and R."""
    edges = []
    for l in L:
        for r in R:
            e = (min(l, r), max(l, r))
            edges.append(e)
    return edges

def _greedy_partition(adj, edges_list, n, deadline):
    """Greedy set cover: repeatedly pick the biclique covering most uncovered edges."""
    all_edges = set()
    for u, v in edges_list:
        all_edges.add((min(u, v), max(u, v)))
    
    uncovered = set(all_edges)
    bicliques = []
    
    # Precompute edge list sorted for deterministic ordering
    edge_list = sorted(all_edges)
    
    while uncovered:
        if time.time() > deadline:
            break
        
        best_biclique = None
        best_count = 0
        best_edges = set()
        
        # Try each uncovered edge as a seed
        for (su, sv) in list(uncovered):
            if time.time() > deadline:
                break
            
            result = _grow_biclique(su, sv, adj, uncovered, n)
            if result is None:
                continue
            L, R = result
            bic_edges = _biclique_edges(L, R)
            bic_set = set(bic_edges)
            # Only count uncovered edges
            covered = bic_set & uncovered
            if len(covered) > best_count:
                best_count = len(covered)
                best_biclique = (L, R)
                best_edges = bic_set
        
        if best_biclique is None or best_count == 0:
            # Fallback: single edge
            e = next(iter(uncovered))
            L, R = {e[0]}, {e[1]}
            best_biclique = (L, R)
            best_edges = {e}
        
        L, R = best_biclique
        bicliques.append((L, R))
        uncovered -= best_edges
    
    return bicliques

def _try_merge(bicliques, adj, edges_set, n):
    """Try to merge two bicliques into one if their union is still a valid biclique."""
    changed = True
    while changed:
        changed = False
        for i in range(len(bicliques)):
            if changed:
                break
            for j in range(i + 1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                # Try merging: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                # But we need all cross pairs to be edges
                new_L = L1 | L2
                new_R = R1 | R2
                # Check: all (l, r) for l in new_L, r in new_R must be edges
                valid = True
                for l in new_L:
                    for r in new_R:
                        if (min(l, r), max(l, r)) not in edges_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that the merged biclique's edge set is exactly the union
                    merged_edges = _biclique_edges(new_L, new_R)
                    merged_set = set(merged_edges)
                    old_edges = set(_biclique_edges(L1, R1)) | set(_biclique_edges(L2, R2))
                    if merged_set == old_edges:
                        bicliques[i] = (new_L, new_R)
                        bicliques.pop(j)
                        changed = True
                        break
    return bicliques

def _trim_bicliques(bicliques, adj, edges_set, n):
    """Remove vertices from bicliques if they don't contribute unique edges."""
    for i in range(len(bicliques)):
        L, R = bicliques[i]
        # For each vertex in L, check if removing it still leaves all its edges covered by other bicliques
        # This is complex; skip for now
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    n = P['n']
    edges_list = P['edges']
    adj = _build_adj(n, edges_list)
    
    all_edges = set()
    for u, v in edges_list:
        all_edges.add((min(u, v), max(u, v)))
    
    # Run greedy partition
    bicliques = _greedy_partition(adj, edges_list, n, deadline)
    
    # Try to merge
    if time.time() < deadline:
        bicliques = _try_merge(bicliques, adj, all_edges, n)
    
    # Validate: check that all edges are covered exactly once
    covered_count = [0] * (n * n)
    valid = True
    for L, R in bicliques:
        if not L or not R:
            valid = False
            break
        if L & R:
            valid = False
            break
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e not in all_edges:
                    valid = False
                    break
                idx = e[0] * n + e[1]
                covered_count[idx] += 1
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to seed solution
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    # Check all edges covered exactly once
    for e in all_edges:
        idx = e[0] * n + e[1]
        if covered_count[idx] != 1:
            valid = False
            break
    
    if not valid:
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    # Convert to required format: list of [left_list, right_list]
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result

# EVOLVE-BLOCK-END
