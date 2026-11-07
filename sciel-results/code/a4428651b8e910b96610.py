import time
import itertools
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build data structures
    adj_matrix = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj_matrix[u][v] = True
        adj_matrix[v][u] = True
        edge_set.add(_edge_key(u, v))
        
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    # Greedy Construction with multiple restarts
    best_bicliques = None
    best_count = float('inf')
    
    num_restarts = 15
    per_restart_time = min(4.0, (deadline - start_time - 5.0) / max(1, num_restarts))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
            
        remaining_edges = set(edge_set)
        bicliques = []
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                break
                
            candidates = list(remaining_edges)
            if not candidates:
                break
            
            # Sort by degree sum, with some randomness for diversity
            candidates.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
            
            # Try top K candidates with some randomness
            k = min(8, len(candidates))
            if restart > 0:
                # Add randomness to candidate selection
                top_candidates = candidates[:max(k, 5)]
                if len(top_candidates) > 1:
                    random.shuffle(top_candidates)
                candidates = top_candidates
            
            best_biclique = None
            best_covered = -1
            
            for start_edge in candidates[:k]:
                if time.time() > deadline - 3.0:
                    break
                u, v = start_edge
                L = {u}
                R = {v}
                
                # Expand L and R iteratively
                changed = True
                max_iters = 100
                iter_count = 0
                while changed and iter_count < max_iters:
                    changed = False
                    iter_count += 1
                    
                    # Try adding to L: w can join L if all (w,x) for x in R are in remaining_edges
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        can_add = True
                        for x in R:
                            if _edge_key(w, x) not in remaining_edges:
                                can_add = False
                                break
                        if can_add:
                            L.add(w)
                            changed = True
                    
                    # Try adding to R: w can join R if all (w,x) for x in L are in remaining_edges
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        can_add = True
                        for x in L:
                            if _edge_key(w, x) not in remaining_edges:
                                can_add = False
                                break
                        if can_add:
                            R.add(w)
                            changed = True
                
                # Validate and compute cross edges
                cross_edges = set()
                valid_biclique = True
                for u_l in L:
                    for v_r in R:
                        k_edge = _edge_key(u_l, v_r)
                        if k_edge not in remaining_edges:
                            valid_biclique = False
                            break
                        cross_edges.add(k_edge)
                    if not valid_biclique:
                        break
                
                if valid_biclique and len(cross_edges) > best_covered:
                    best_covered = len(cross_edges)
                    best_biclique = (frozenset(L), frozenset(R), cross_edges)
            
            if best_biclique is None:
                u, v = candidates[0]
                bicliques.append(({u}, {v}))
                remaining_edges.remove(_edge_key(u, v))
            else:
                L, R, covered = best_biclique
                bicliques.append((L, R))
                remaining_edges -= covered
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques[:]
        
        if best_count <= 6:
            break
    
    if best_bicliques is None:
        # Fallback
        result = []
        for u, v in edges:
            result.append([[u], [v]])
        return result
    
    # Post-Processing: Multiple merge passes
    current_bicliques = [(list(L), list(R)) for L, R in best_bicliques]
    
    for pass_num in range(20):
        if time.time() > deadline - 3.0:
            break
            
        merged = False
        for i in range(len(current_bicliques)):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, len(current_bicliques)):
                if time.time() > deadline - 2.0:
                    break
                
                Li, Ri = current_bicliques[i]
                Lj, Rj = current_bicliques[j]
                
                combos = [
                    (set(Li) | set(Lj), set(Ri) | set(Rj)),
                    (set(Li) | set(Rj), set(Ri) | set(Lj)),
                    (set(Ri) | set(Lj), set(Li) | set(Rj)),
                    (set(Ri) | set(Rj), set(Li) | set(Lj))
                ]
                
                for NL, NR in combos:
                    if not NL or not NR:
                        continue
                    if NL & NR:
                        continue
                    
                    valid = True
                    cross_edges = set()
                    for u in NL:
                        for v in NR:
                            k_edge = _edge_key(u, v)
                            if k_edge not in edge_set:
                                valid = False
                                break
                            cross_edges.add(k_edge)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    cross_i = set()
                    for u in Li:
                        for v in Ri:
                            cross_i.add(_edge_key(u, v))
                            
                    cross_j = set()
                    for u in Lj:
                        for v in Rj:
                            cross_j.add(_edge_key(u, v))
                            
                    if cross_edges == (cross_i | cross_j):
                        current_bicliques[i] = (list(NL), list(NR))
                        current_bicliques.pop(j)
                        merged = True
                        break
            if merged:
                break
        
        if not merged:
            break
    
    # Local search: try to move vertices between bicliques to reduce count
    for ls_iter in range(50):
        if time.time() > deadline - 2.0:
            break
            
        improved = False
        for i in range(len(current_bicliques)):
            if time.time() > deadline - 1.5:
                break
            for j in range(len(current_bicliques)):
                if i == j:
                    continue
                if time.time() > deadline - 1.5:
                    break
                    
                Li, Ri = current_bicliques[i]
                Lj, Rj = current_bicliques[j]
                
                # Try moving a vertex from Li to Lj or Ri to Rj, etc.
                # This is complex, so skip for now and just do final validation
        
        if not improved:
            break
    
    # Final Validation and Formatting
    covered = set()
    valid = True
    for L, R in current_bicliques:
        if not L or not R:
            valid = False
            break
        if set(L) & set(R):
            valid = False
            break
        for u in L:
            for v in R:
                k_edge = _edge_key(u, v)
                if k_edge not in edge_set:
                    valid = False
                    break
                if k_edge in covered:
                    valid = False
                    break
                covered.add(k_edge)
    if not valid or covered != edge_set:
        result = []
        for u, v in edges:
            result.append([[u], [v]])
        return result
    
    result = []
    for L, R in current_bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
        
    return result
