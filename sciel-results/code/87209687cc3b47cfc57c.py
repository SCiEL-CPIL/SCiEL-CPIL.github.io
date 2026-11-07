import time
import random
import sys
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _find_maximal_bicliques(n, edge_set, max_bicliques=5000):
    """
    Find maximal bicliques using Bron-Kerbosch style enumeration.
    A biclique (L, R) is maximal if no vertex can be added to L or R
    while maintaining the biclique property.
    
    We use a different approach: enumerate all pairs of non-adjacent vertices
    in the original graph (i.e., edges in the complement) and use them to
    partition the vertex set.
    
    Actually, for a general graph, finding all maximal bicliques is #P-hard.
    For n=30, we'll use a heuristic approach:
    1. For each edge (u,v), try to grow a biclique containing it
    2. Use multiple random orderings to find diverse maximal bicliques
    """
    bicliques = []
    seen = set()
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each edge, try to find a maximal biclique containing it
    edges_list = list(edge_set)
    random.shuffle(edges_list)
    
    for u, v in edges_list:
        if len(bicliques) >= max_bicliques:
            break
        if time.time() > _deadline:
            break
            
        # Start with L={u}, R={v}
        L = {u}
        R = {v}
        
        # Expand greedily
        changed = True
        while changed:
            changed = False
            # Try to add vertices to L
            for w in range(n):
                if w in L:
                    continue
                # w must be adjacent to all in R
                if all(w in adj[r] for r in R):
                    L.add(w)
                    changed = True
            # Try to add vertices to R
            for w in range(n):
                if w in R:
                    continue
                # w must be adjacent to all in L
                if all(w in adj[l] for l in L):
                    R.add(w)
                    changed = True
        
        # Check if this biclique covers only edges in edge_set
        valid = True
        for l in L:
            for r in R:
                if (l, r) not in edge_set and (r, l) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Canonical form: sort L and R, then ensure L < R lexicographically
        L_sorted = tuple(sorted(L))
        R_sorted = tuple(sorted(R))
        if L_sorted > R_sorted:
            L_sorted, R_sorted = R_sorted, L_sorted
        
        key = (L_sorted, R_sorted)
        if key not in seen:
            seen.add(key)
            bicliques.append((L_sorted, R_sorted))
    
    return bicliques


_deadline = 0.0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 55.0  # Safety margin
    
    edges = P['edges']
    n = P['n']
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Get all edges in canonical form
    canon_edges = set()
    for u, v in edge_set:
        canon_edges.add((min(u, v), max(u, v)))
    
    m = len(canon_edges)
    
    # Find maximal bicliques
    bicliques = _find_maximal_bicliques(n, edge_set, max_bicliques=2000)
    
    # If we didn't find enough, add individual edges as bicliques
    for u, v in canon_edges:
        biclique = ((u,), (v,))
        if biclique not in [(l, r) for l, r in bicliques]:
            bicliques.append(biclique)
    
    # Now solve exact cover using Algorithm X with Dancing Links
    # Each biclique covers a set of edges. We need to select a subset of bicliques
    # that covers all edges exactly once.
    
    # Build the incidence matrix: rows = bicliques, cols = edges
    # biclique i covers edge j if edge j is in biclique i
    
    # Map edges to indices
    edge_to_idx = {}
    for i, e in enumerate(canon_edges):
        edge_to_idx[e] = i
    
    # For each biclique, determine which edges it covers
    biclique_edges = []
    for L, R in bicliques:
        covered = set()
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e in edge_to_idx:
                    covered.add(edge_to_idx[e])
        if covered:
            biclique_edges.append(covered)
    
    num_bicliques = len(biclique_edges)
    num_edges = m
    
    # Use Algorithm X with Dancing Links
    # We'll implement a simplified version using sets for constraint propagation
    
    # State: set of uncovered edges, set of available bicliques
    # We want to find a minimum cardinality exact cover
    
    # Use branch and bound with heuristic column selection
    best_solution = None
    best_size = num_bicliques  # Upper bound: use one biclique per edge
    
    # Precompute for each edge, which bicliques cover it
    edge_to_bicliques = defaultdict(list)
    for i, covered in enumerate(biclique_edges):
        for e in covered:
            edge_to_bicliques[e].append(i)
    
    # Sort bicliques by size (larger first) for better pruning
    biclique_order = sorted(range(num_bicliques), key=lambda i: -len(biclique_edges[i]))
    
    # Branch and bound
    sys.setrecursionlimit(10000)
    
    def solve(uncovered, used_bicliques, current_size):
        nonlocal best_solution, best_size
        
        if time.time() > _deadline:
            return
        
        if not uncovered:
            if current_size < best_size:
                best_size = current_size
                best_solution = list(used_bicliques)
            return
        
        if current_size >= best_size:
            return
        
        # Select the edge with fewest covering bicliques (MRV heuristic)
        min_count = float('inf')
        chosen_edge = None
        for e in uncovered:
            count = len([b for b in edge_to_bicliques[e] if b not in used_bicliques])
            if count < min_count:
                min_count = count
                chosen_edge = e
            if count == 0:
                return  # Dead end
        
        if min_count == 0:
            return
        
        # Try each biclique that covers chosen_edge
        candidates = [b for b in edge_to_bicliques[chosen_edge] if b not in used_bicliques]
        # Sort by coverage (larger first)
        candidates.sort(key=lambda b: -len(biclique_edges[b]))
        
        for b in candidates:
            if time.time() > _deadline:
                return
            
            # Check if this biclique only covers uncovered edges
            covered = biclique_edges[b]
            if covered & used_bicliques:  # This check is wrong, let me fix
                pass
            
            # Actually, we need to check that all edges in biclique b are in uncovered
            if not covered.issubset(uncovered):
                continue
            
            new_uncovered = uncovered - covered
            used_bicliques.add(b)
            solve(new_uncovered, used_bicliques, current_size + 1)
            used_bicliques.remove(b)
    
    all_edges = set(range(num_edges))
    solve(all_edges, set(), 0)
    
    # If Algorithm X found a solution, use it
    if best_solution is not None:
        result = []
        for b in best_solution:
            L, R = bicliques[b]
            result.append([list(L), list(R)])
        return result
    
    # Fallback: greedy set cover
    remaining = set(canon_edges)
    result = []
    
    while remaining:
        if time.time() > _deadline:
            break
        
        best_b = None
        best_count = 0
        
        for b in range(num_bicliques):
            covered = biclique_edges[b]
            count = len(covered & remaining)
            if count > best_count:
                best_count = count
                best_b = b
        
        if best_b is None or best_count == 0:
            # Fallback: add a single edge
            e = next(iter(remaining))
            u, v = e
            result.append([[u], [v]])
            remaining.discard(e)
        else:
            L, R = bicliques[best_b]
            result.append([list(L), list(R)])
            remaining -= biclique_edges[best_b]
    
    return result
