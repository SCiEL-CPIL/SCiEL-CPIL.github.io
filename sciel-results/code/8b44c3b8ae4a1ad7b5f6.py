import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _expand_biclique(L, R, adj, n, deadline):
    """Maximally expand a biclique (L, R) using the given adjacency."""
    L = set(L)
    R = set(R)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Try to add vertices to L: w must be adjacent to all in R
        if R:
            common = set.intersection(*(adj[r] for r in R))
            for w in common:
                if w not in L:
                    L.add(w)
                    changed = True
        # Try to add vertices to R: w must be adjacent to all in L
        if L:
            common = set.intersection(*(adj[l] for l in L))
            for w in common:
                if w not in R:
                    R.add(w)
                    changed = True
    return L, R

def _count_covered(L, R, remaining):
    """Count how many edges in remaining are covered by biclique (L, R)."""
    count = 0
    for l in L:
        for r in R:
            if _edge_key(l, r) in remaining:
                count += 1
    return count

def _weighted_biclique_search(remaining, adj, n, edge_weights, deadline):
    """Find maximum-weight biclique in residual edge set using anchor-based search.
    
    edge_weights maps edge_key -> weight (higher = more important to cover)
    Returns (L, R) of best biclique found, or None if no edges remain.
    """
    if not remaining:
        return None
    
    # Build list of candidate anchor edges sorted by total weight
    anchor_edges = []
    for e in remaining:
        w = edge_weights.get(e, 1.0)
        u, v = e
        # Use degree sum as secondary sort key
        deg_sum = len(adj[u]) + len(adj[v])
        anchor_edges.append((w, deg_sum, e))
    
    anchor_edges.sort(reverse=True)
    
    best_L = None
    best_R = None
    best_weight = 0.0
    
    # Try top anchors
    num_anchors = min(60, len(anchor_edges))
    for i in range(num_anchors):
        if time.time() > deadline:
            break
        _, _, e = anchor_edges[i]
        u, v = e
        
        # Try both orientations
        for (a, b) in [(u, v), (v, u)]:
            L = {a}
            R = {b}
            L, R = _expand_biclique(L, R, adj, n, deadline)
            
            # Calculate total weight of this biclique
            weight = 0.0
            for l in L:
                for r in R:
                    key = _edge_key(l, r)
                    if key in remaining:
                        weight += edge_weights.get(key, 1.0)
            
            if weight > best_weight:
                best_weight = weight
                best_L = L
                best_R = R
    
    if best_L is not None:
        return best_L, best_R
    return None

def _greedy_partition(edge_set, n, deadline, seed, num_starts=50):
    """Greedy biclique extraction with weighted pricing (column-generation style)."""
    rng = random.Random(seed)
    remaining = set(edge_set)
    adj = _build_adjacency(remaining, n)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Compute edge weights: inverse frequency of how many times an edge 
        # could be covered by potential bicliques (approximated by 1/degree_sum)
        edge_weights = {}
        for e in remaining:
            u, v = e
            deg_sum = len(adj[u]) + len(adj[v])
            # Higher weight for edges with fewer alternative coverings
            edge_weights[e] = 1.0 / max(1, deg_sum)
        
        # Use weighted biclique search
        result = _weighted_biclique_search(remaining, adj, n, edge_weights, deadline)
        
        if result is not None:
            L, R = result
        else:
            # Fallback: pick a random edge
            u, v = next(iter(remaining))
            L, R = _expand_biclique({u}, {v}, adj, n, deadline)
        
        bicliques.append([list(L), list(R)])
        
        # Remove covered edges
        for l in L:
            for r in R:
                key = _edge_key(l, r)
                if key in remaining:
                    remaining.discard(key)
                    adj[l].discard(r)
                    adj[r].discard(l)
    
    return bicliques

def _try_merge(bicliques, edge_set, n, deadline):
    """Try to merge pairs of bicliques if their union forms a valid biclique."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                L1, R1 = set(bicliques[i][0]), set(bicliques[i][1])
                L2, R2 = set(bicliques[j][0]), set(bicliques[j][1])
                
                # Try merging: (L1∪L2, R1∩R2)
                L_new = L1 | L2
                R_new = R1 & R2
                if L_new and R_new:
                    valid = True
                    for l in L_new:
                        for r in R_new:
                            if _edge_key(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Check coverage: all edges from biclique i and j must be in merged
                        all_covered = True
                        for l in L1:
                            for r in R1:
                                if _edge_key(l, r) in edge_set:
                                    if l not in L_new or r not in R_new:
                                        all_covered = False
                                        break
                            if not all_covered:
                                break
                        if all_covered:
                            for l in L2:
                                for r in R2:
                                    if _edge_key(l, r) in edge_set:
                                        if l not in L_new or r not in R_new:
                                            all_covered = False
                                            break
                                    if not all_covered:
                                        break
                        if all_covered:
                            bicliques[i] = [list(L_new), list(R_new)]
                            bicliques.pop(j)
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break
    return bicliques

def _local_search(bicliques, edge_set, n, deadline):
    """Local search: try to remove redundant bicliques and shrink bicliques."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to remove a biclique if its edges are already covered by others
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            L, R = set(bicliques[i][0]), set(bicliques[i][1])
            # Check if all edges in this biclique are covered by other bicliques
            all_covered = True
            for l in L:
                for r in R:
                    key = _edge_key(l, r)
                    if key in edge_set:
                        # Check if any other biclique covers this edge
                        covered_by_other = False
                        for j in range(len(bicliques)):
                            if j == i:
                                continue
                            Lj, Rj = set(bicliques[j][0]), set(bicliques[j][1])
                            if l in Lj and r in Rj:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            all_covered = False
                            break
                if not all_covered:
                    break
            if all_covered:
                bicliques.pop(i)
                improved = True
                break
        if improved:
            continue
        
        # Try to shrink a biclique: remove a vertex from L or R if it doesn't lose coverage
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            L, R = set(bicliques[i][0]), set(bicliques[i][1])
            
            # Try removing each vertex from L
            for l in list(L):
                if time.time() > deadline:
                    break
                L_new = L - {l}
                if not L_new:
                    continue
                # Check: does removing l lose any edge coverage?
                # An edge (l, r) for r in R must be covered by some other biclique
                can_remove = True
                for r in R:
                    key = _edge_key(l, r)
                    if key in edge_set:
                        covered = False
                        for j in range(len(bicliques)):
                            if j == i:
                                continue
                            Lj, Rj = set(bicliques[j][0]), set(bicliques[j][1])
                            if l in Lj and r in Rj:
                                covered = True
                                break
                        if not covered:
                            can_remove = False
                            break
                if can_remove:
                    bicliques[i] = [list(L_new), list(R)]
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing each vertex from R
            for r in list(R):
                if time.time() > deadline:
                    break
                R_new = R - {r}
                if not R_new:
                    continue
                can_remove = True
                for l in L:
                    key = _edge_key(l, r)
                    if key in edge_set:
                        covered = False
                        for j in range(len(bicliques)):
                            if j == i:
                                continue
                            Lj, Rj = set(bicliques[j][0]), set(bicliques[j][1])
                            if l in Lj and r in Rj:
                                covered = True
                                break
                        if not covered:
                            can_remove = False
                            break
                if can_remove:
                    bicliques[i] = [list(L), list(R_new)]
                    improved = True
                    break
            
            if improved:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # Safety margin before 90s limit
    
    edges = P['edges']
    n = P['n']
    edge_set = set(_edge_key(u, v) for u, v in edges)
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multiple random restarts with different seeds
    num_restarts = 40
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        bicliques = _greedy_partition(edge_set, n, deadline, seed, num_starts=40)
        
        # Try to merge bicliques to reduce count
        bicliques = _try_merge(bicliques, edge_set, n, deadline)
        
        # Local search to remove redundant bicliques and shrink
        bicliques = _local_search(bicliques, edge_set, n, deadline)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    if best_bicliques is None:
        # Fallback: star partition
        adj = _build_adjacency(edge_set, n)
        bicliques = []
        for v in range(n):
            if adj[v]:
                bicliques.append([[v], list(adj[v])])
        best_bicliques = bicliques
    
    # Final validation and cleanup: ensure proper types
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    
    return result
