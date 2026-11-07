import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _build_adj(n, edge_set):
    """Build adjacency as list of sets."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _expand_biclique(start_edge, adj, n, deadline):
    """
    Expand a biclique starting from start_edge to a maximal biclique.
    Returns (L, R) as sets.
    """
    u, v = start_edge
    L = {u}
    R = {v}
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Try to add to L: w can join L if connected to all in R
        for w in range(n):
            if w not in L and w not in R:
                if all(r in adj[w] for r in R):
                    L.add(w)
                    changed = True
        
        # Try to add to R: w can join R if connected to all in L
        for w in range(n):
            if w not in R and w not in L:
                if all(l in adj[w] for l in L):
                    R.add(w)
                    changed = True
    
    return L, R


def _greedy_partition(edge_set, n, adj, deadline, seed=None):
    """
    Greedy set cover: repeatedly pick the biclique covering most uncovered edges.
    """
    if seed is not None:
        random.seed(seed)
    
    bicliques = []
    remaining = set(edge_set)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Compute degrees in remaining subgraph
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        # Build adjacency for remaining subgraph
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Generate candidates: top-degree edges + random
        sorted_edges = sorted(remaining, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        candidates = sorted_edges[:15]
        if len(remaining) > 20:
            candidates.extend(random.sample(list(remaining), min(10, len(remaining))))
        
        best_biclique = None
        best_cover = 0
        
        for edge in candidates:
            if time.time() > deadline:
                break
            L, R = _expand_biclique(edge, rem_adj, n, deadline)
            
            # Count actual edges covered
            cover = 0
            for l in L:
                for r in R:
                    if _edge_key(l, r) in remaining:
                        cover += 1
            
            if cover > best_cover:
                best_cover = cover
                best_biclique = (L, R)
        
        if best_biclique is None or best_cover == 0:
            edge = next(iter(remaining))
            u, v = edge
            best_biclique = ({u}, {v})
        
        L, R = best_biclique
        bicliques.append((L, R))
        
        # Remove covered edges
        for l in L:
            for r in R:
                ek = _edge_key(l, r)
                if ek in remaining:
                    remaining.remove(ek)
    
    # Leftover edges as single-edge bicliques
    for u, v in remaining:
        bicliques.append(({u}, {v}))
    
    return bicliques


def _try_merge_pass(bicliques, edge_set, n, deadline):
    """
    Try to merge pairs of bicliques if their union forms a valid biclique
    that covers exactly the union of their edges.
    """
    current = [(set(L), set(R)) for L, R in bicliques]
    changed = True
    
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current)):
                if time.time() > deadline:
                    break
                
                Li, Ri = current[i]
                Lj, Rj = current[j]
                
                # Try different merge orientations
                merges = [
                    (Li | Lj, Ri | Rj),
                    (Li | Rj, Ri | Lj),
                    (Ri | Lj, Li | Rj),
                    (Ri | Rj, Li | Lj),
                ]
                
                for newL, newR in merges:
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue
                    
                    # Check: all cross pairs must be edges in original graph
                    valid = True
                    covered_edges = set()
                    for l in newL:
                        for r in newR:
                            ek = _edge_key(l, r)
                            if ek not in edge_set:
                                valid = False
                                break
                            covered_edges.add(ek)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check: covered_edges must equal union of edges from biclique i and j
                    edges_i = set()
                    for l in Li:
                        for r in Ri:
                            edges_i.add(_edge_key(l, r))
                    edges_j = set()
                    for l in Lj:
                        for r in Rj:
                            edges_j.add(_edge_key(l, r))
                    
                    if covered_edges == (edges_i | edges_j) and not (edges_i & edges_j):
                        # Valid merge
                        current.pop(j)
                        current[i] = (newL, newR)
                        changed = True
                        break
                
                if changed:
                    break
            
            if changed:
                break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build edge set and adjacency
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    adj = _build_adj(n, edge_set)
    
    best_solution = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        bicliques = _greedy_partition(edge_set, n, adj, deadline - 5.0, seed=seed)
        count = len(bicliques)
        
        if count < best_count:
            best_count = count
            best_solution = bicliques[:]
        
        if best_count <= 8:
            break
    
    if best_solution is None:
        # Fallback
        best_solution = [({u}, {v}) for u, v in edge_set]
    
    # Phase 2: Merge passes to reduce count
    merged = _try_merge_pass(best_solution, edge_set, n, deadline - 3.0)
    
    if len(merged) < best_count:
        best_count = len(merged)
        best_solution = merged
    
    # Phase 3: Try a few more merge passes with different initial orderings
    for attempt in range(5):
        if time.time() > deadline - 2.0:
            break
        
        # Shuffle bicliques to find different merge opportunities
        shuffled = best_solution[:]
        random.shuffle(shuffled)
        merged2 = _try_merge_pass(shuffled, edge_set, n, deadline - 1.0)
        
        if len(merged2) < best_count:
            best_count = len(merged2)
            best_solution = merged2
    
    # Convert to required format: list of [left_list, right_list]
    result = []
    for L, R in best_solution:
        result.append([list(L), list(R)])
    
    return result
