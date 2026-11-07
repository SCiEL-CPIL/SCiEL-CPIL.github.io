"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(e):
    a, b = e
    if a > b:
        a, b = b, a
    return (a, b)

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_biclique(adj, remaining_edges, n):
    """Find a large biclique using common-neighbor heuristic on remaining edges."""
    if not remaining_edges:
        return None, set()
    
    # Convert to set for O(1) lookup
    rem_set = set(remaining_edges)
    
    best_L = set()
    best_R = set()
    best_edges = 0
    
    # Limit seed candidates to keep it fast
    seeds = list(rem_set)
    if len(seeds) > 40:
        step = len(seeds) / 40.0
        idxs = set()
        for i in range(40):
            idxs.add(int(i * step))
        idxs.add(0)
        idxs.add(len(seeds) - 1)
        seeds = [seeds[i] for i in sorted(idxs)]
    
    for u, v in seeds:
        L = {u}
        R = {v}
        
        for _round in range(8):
            # potential L additions: vertices x (not in L) such that x connected in remaining to all in R
            cand_L = []
            for x in range(n):
                if x in L:
                    continue
                ok = True
                for r in R:
                    if (min(x, r), max(x, r)) not in rem_set:
                        ok = False
                        break
                if ok:
                    cand_L.append(x)
            
            # potential R additions: vertices y (not in R) such that y connected in remaining to all in L
            cand_R = []
            for y in range(n):
                if y in R:
                    continue
                ok = True
                for l in L:
                    if (min(l, y), max(l, y)) not in rem_set:
                        ok = False
                        break
                if ok:
                    cand_R.append(y)
            
            if not cand_L and not cand_R:
                break
            
            # Choose candidate that adds most remaining edges
            # Gain for L candidate x: |R|
            # Gain for R candidate y: |L|
            if cand_L and (not cand_R or len(R) >= len(L)):
                # Pick x in cand_L with max remaining degree (tie-breaker)
                x = max(cand_L, key=lambda z: len([e for e in rem_set if z in e]))
                L.add(x)
            elif cand_R:
                y = max(cand_R, key=lambda z: len([e for e in rem_set if z in e]))
                R.add(y)
            else:
                break
        
        # Count edges covered
        cnt = 0
        for l in L:
            for r in R:
                if (min(l, r), max(l, r)) in rem_set:
                    cnt += 1
        
        if cnt > best_edges:
            best_edges = cnt
            best_L = set(L)
            best_R = set(R)
    
    if best_edges == 0:
        u, v = next(iter(rem_set))
        return ({u}, {v}), {(min(u, v), max(u, v))}
    
    covered = set()
    for l in best_L:
        for r in best_R:
            k = (min(l, r), max(l, r))
            if k in rem_set:
                covered.add(k)
    
    return (best_L, best_R), covered

def construct():
    start = time.time()
    deadline = start + 75.0
    n = P['n']
    edges = P['edges']
    
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add((min(u, v), max(u, v)))
    
    adj = _build_adj(n, edges)
    remaining = set(all_edges_set)
    bicliques = []
    
    # Phase 1: Greedy construction
    while remaining:
        if time.time() > deadline:
            break
        (L, R), covered = _find_max_biclique(adj, remaining, n)
        if not covered:
            k = next(iter(remaining))
            L, R = {k[0]}, {k[1]}
            covered = {k}
        bicliques.append((L, R))
        remaining -= covered
    
    # If we ran out of time, complete trivially with single-edge bicliques
    if remaining:
        for k in list(remaining):
            bicliques.append(({k[0]}, {k[1]}))
        remaining = set()
    
    # Phase 2: Local search - merge bicliques if possible
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merging: L = L1|L2, R = R1|R2
                L = set(L1) | set(L2)
                R = set(R1) | set(R2)
                
                if L & R:
                    continue
                
                # Check if every cross pair is an edge in the original graph
                ok = True
                for l in L:
                    for r in R:
                        if (min(l, r), max(l, r)) not in all_edges_set:
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    # Ensure the union's cross edges exactly match the union of the two bicliques' cross edges
                    new_edges = set()
                    for l in L:
                        for r in R:
                            new_edges.add((min(l, r), max(l, r)))
                    
                    old_edges = set()
                    for l in L1:
                        for r in R1:
                            old_edges.add((min(l, r), max(l, r)))
                    for l in L2:
                        for r in R2:
                            old_edges.add((min(l, r), max(l, r)))
                    
                    if new_edges == old_edges:
                        bicliques[i] = (L, R)
                        bicliques.pop(j)
                        changed = True
                        break
            if changed:
                break
    
    # Phase 3: Try to move vertices between bicliques to enable further merges
    # This is a more sophisticated local search step
    if time.time() < deadline:
        for _iteration in range(3):  # Limit iterations to stay within time
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Try moving a vertex from L1 to R2 or from R1 to L2, etc.
                    # This is complex, so we'll just do a simple version:
                    # Check if we can split a biclique to make it mergeable with another
                    
                    # For now, skip this complex step and rely on the merge phase
                    pass
            
            if not improved:
                break
    
    # Convert to lists of ints
    result = []
    for L, R in bicliques:
        result.append([list(int(x) for x in sorted(L)), list(int(x) for x in sorted(R))])
    
    return result
# EVOLVE-BLOCK-END
