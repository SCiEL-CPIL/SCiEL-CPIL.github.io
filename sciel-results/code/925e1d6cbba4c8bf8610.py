"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start = time.time()
    deadline = start + 78.0  # safety margin
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    all_edges_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        all_edges_set.add((min(u, v), max(u, v)))
    
    remaining = set(all_edges_set)
    bicliques = []
    
    def _cross_edges(L, R):
        s = set()
        for l in L:
            for r in R:
                s.add((min(l, r), max(l, r)))
        return s
    
    def _is_valid_biclique(L, R, edge_set):
        if L & R:
            return False
        for l in L:
            for r in R:
                if (min(l, r), max(l, r)) not in edge_set:
                    return False
        return True
    
    def _expand_biclique(u, v, rem_set):
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Try adding to L
            for x in range(n):
                if x in L or x in R:
                    continue
                if all((min(x, r), max(x, r)) in rem_set for r in R):
                    L.add(x)
                    changed = True
            # Try adding to R
            for y in range(n):
                if y in L or y in R:
                    continue
                if all((min(l, y), max(l, y)) in rem_set for l in L):
                    R.add(y)
                    changed = True
        return L, R
    
    # Phase 1: Greedy set-cover with maximal biclique expansion
    while remaining and time.time() < deadline:
        best_biclique = None
        best_covered = set()
        best_score = -1
        
        rem_list = list(remaining)
        # Use more seeds when few edges remain
        max_seeds = min(len(rem_list), 80) if len(rem_list) > 40 else len(rem_list)
        if len(rem_list) > max_seeds:
            step = len(rem_list) / max_seeds
            idxs = set()
            for i in range(max_seeds):
                idxs.add(int(i * step))
            idxs.add(0)
            idxs.add(len(rem_list) - 1)
            seeds = [rem_list[i] for i in sorted(idxs)]
        else:
            seeds = rem_list
        
        for u, v in seeds:
            L, R = _expand_biclique(u, v, remaining)
            
            if L & R:
                continue
            
            covered = set()
            for l in L:
                for r in R:
                    k = (min(l, r), max(l, r))
                    if k in remaining:
                        covered.add(k)
            
            if not covered:
                continue
            
            # Score: prefer covering more edges; penalize large bicliques slightly
            score = len(covered) / max(1, len(L) + len(R))
            if score > best_score:
                best_score = score
                best_biclique = (L, R)
                best_covered = covered
        
        if best_biclique is None or not best_covered:
            k = next(iter(remaining))
            best_biclique = ({k[0]}, {k[1]})
            best_covered = {k}
        
        bicliques.append(best_biclique)
        remaining -= best_covered
    
    if remaining:
        for k in list(remaining):
            bicliques.append(({k[0]}, {k[1]}))
        remaining = set()
    
    # Phase 2: Try to merge bicliques (direct union, cross-swap, and vertex migration)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                old_e = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                
                # Attempt 1: Direct merge L=L1|L2, R=R1|R2
                L = set(L1) | set(L2)
                R = set(R1) | set(R2)
                if not (L & R):
                    if _is_valid_biclique(L, R, all_edges_set):
                        new_e = _cross_edges(L, R)
                        if new_e == old_e:
                            bicliques[i] = (L, R)
                            bicliques.pop(j)
                            changed = True
                            break
                
                # Attempt 2: Cross-swap L=L1|R2, R=R1|L2
                Ls = set(L1) | set(R2)
                Rs = set(R1) | set(L2)
                if not (Ls & Rs):
                    if _is_valid_biclique(Ls, Rs, all_edges_set):
                        new_e = _cross_edges(Ls, Rs)
                        if new_e == old_e:
                            bicliques[i] = (Ls, Rs)
                            bicliques.pop(j)
                            changed = True
                            break
                
                # Attempt 3: L=R1|L2, R=L1|R2
                Ls2 = set(R1) | set(L2)
                Rs2 = set(L1) | set(R2)
                if not (Ls2 & Rs2):
                    if _is_valid_biclique(Ls2, Rs2, all_edges_set):
                        new_e = _cross_edges(Ls2, Rs2)
                        if new_e == old_e:
                            bicliques[i] = (Ls2, Rs2)
                            bicliques.pop(j)
                            changed = True
                            break
                
                # Attempt 4: Move vertices from smaller biclique into larger to enable merge
                # Try moving each vertex of biclique j into biclique i's appropriate side
                for v in list(L2):
                    new_L2 = L2 - {v}
                    if not new_L2:
                        continue
                    # Move v to L1
                    if _is_valid_biclique(L1 | {v}, R1, all_edges_set) and _is_valid_biclique(new_L2, R2, all_edges_set):
                        mL = L1 | {v} | new_L2
                        mR = R1 | R2
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    # Move v to R1
                    if _is_valid_biclique(L1, R1 | {v}, all_edges_set) and _is_valid_biclique(new_L2, R2, all_edges_set):
                        mL = L1 | new_L2
                        mR = R1 | {v} | R2
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
                for v in list(R2):
                    new_R2 = R2 - {v}
                    if not new_R2:
                        continue
                    # Move v to L1
                    if _is_valid_biclique(L1 | {v}, R1, all_edges_set) and _is_valid_biclique(L2, new_R2, all_edges_set):
                        mL = L1 | {v} | L2
                        mR = R1 | new_R2
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    # Move v to R1
                    if _is_valid_biclique(L1, R1 | {v}, all_edges_set) and _is_valid_biclique(L2, new_R2, all_edges_set):
                        mL = L1 | L2
                        mR = R1 | {v} | new_R2
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
                # Also try moving from i to j (reverse direction)
                for v in list(L1):
                    new_L1 = L1 - {v}
                    if not new_L1:
                        continue
                    if _is_valid_biclique(new_L1, R1, all_edges_set) and _is_valid_biclique(L2 | {v}, R2, all_edges_set):
                        mL = new_L1 | L2 | {v}
                        mR = R1 | R2
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    if _is_valid_biclique(new_L1, R1, all_edges_set) and _is_valid_biclique(L2, R2 | {v}, all_edges_set):
                        mL = new_L1 | L2
                        mR = R1 | R2 | {v}
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
                for v in list(R1):
                    new_R1 = R1 - {v}
                    if not new_R1:
                        continue
                    if _is_valid_biclique(L1, new_R1, all_edges_set) and _is_valid_biclique(L2 | {v}, R2, all_edges_set):
                        mL = L1 | L2 | {v}
                        mR = new_R1 | R2
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    if _is_valid_biclique(L1, new_R1, all_edges_set) and _is_valid_biclique(L2, R2 | {v}, all_edges_set):
                        mL = L1 | L2
                        mR = new_R1 | R2 | {v}
                        if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                            ne = _cross_edges(mL, mR)
                            if ne == old_e:
                                bicliques[i] = (mL, mR)
                                bicliques.pop(j)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
            
            if changed:
                break
    
    # Phase 3: Aggressive vertex migration to enable merges
    if time.time() < deadline:
        for _iter in range(12):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(bicliques)):
                    if time.time() > deadline:
                        break
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    old_e = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                    
                    # Try moving a vertex from L1 to L2
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        if not new_L1:
                            continue
                        if _is_valid_biclique(new_L1, R1, all_edges_set) and _is_valid_biclique(L2 | {v}, R2, all_edges_set):
                            mL = new_L1 | L2 | {v}
                            mR = R1 | R2
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[i] = (mL, mR)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                        # Also try cross-merge after move
                        if _is_valid_biclique(new_L1, R1, all_edges_set) and _is_valid_biclique(L2 | {v}, R2, all_edges_set):
                            mL2 = new_L1 | R2
                            mR2 = R1 | L2 | {v}
                            if not (mL2 & mR2) and _is_valid_biclique(mL2, mR2, all_edges_set):
                                ne = _cross_edges(mL2, mR2)
                                if ne == old_e:
                                    bicliques[i] = (mL2, mR2)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                    if improved:
                        break
                    # Try moving a vertex from R1 to R2
                    for v in list(R1):
                        new_R1 = R1 - {v}
                        if not new_R1:
                            continue
                        if _is_valid_biclique(L1, new_R1, all_edges_set) and _is_valid_biclique(L2, R2 | {v}, all_edges_set):
                            mL = L1 | L2
                            mR = new_R1 | R2 | {v}
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[i] = (mL, mR)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                        if _is_valid_biclique(L1, new_R1, all_edges_set) and _is_valid_biclique(L2, R2 | {v}, all_edges_set):
                            mL2 = L1 | R2 | {v}
                            mR2 = new_R1 | L2
                            if not (mL2 & mR2) and _is_valid_biclique(mL2, mR2, all_edges_set):
                                ne = _cross_edges(mL2, mR2)
                                if ne == old_e:
                                    bicliques[i] = (mL2, mR2)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                    if improved:
                        break
                    # Try moving a vertex from L1 to R2 (side swap)
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        if not new_L1:
                            continue
                        if _is_valid_biclique(new_L1, R1, all_edges_set) and _is_valid_biclique(L2, R2 | {v}, all_edges_set):
                            mL = new_L1 | L2
                            mR = R1 | R2 | {v}
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[i] = (mL, mR)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                            mL2 = new_L1 | R2 | {v}
                            mR2 = R1 | L2
                            if not (mL2 & mR2) and _is_valid_biclique(mL2, mR2, all_edges_set):
                                ne = _cross_edges(mL2, mR2)
                                if ne == old_e:
                                    bicliques[i] = (mL2, mR2)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                    if improved:
                        break
                    # Try moving a vertex from R1 to L2 (side swap)
                    for v in list(R1):
                        new_R1 = R1 - {v}
                        if not new_R1:
                            continue
                        if _is_valid_biclique(L1, new_R1, all_edges_set) and _is_valid_biclique(L2 | {v}, R2, all_edges_set):
                            mL = L1 | L2 | {v}
                            mR = new_R1 | R2
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[i] = (mL, mR)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                            mL2 = L1 | R2
                            mR2 = new_R1 | L2 | {v}
                            if not (mL2 & mR2) and _is_valid_biclique(mL2, mR2, all_edges_set):
                                ne = _cross_edges(mL2, mR2)
                                if ne == old_e:
                                    bicliques[i] = (mL2, mR2)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                    if improved:
                        break
                    # Reverse: move from j to i
                    for v in list(L2):
                        new_L2 = L2 - {v}
                        if not new_L2:
                            continue
                        if _is_valid_biclique(new_L2, R2, all_edges_set) and _is_valid_biclique(L1 | {v}, R1, all_edges_set):
                            mL = L1 | {v} | new_L2
                            mR = R1 | R2
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[i] = (mL, mR)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                        if _is_valid_biclique(new_L2, R2, all_edges_set) and _is_valid_biclique(L1 | {v}, R1, all_edges_set):
                            mL2 = L1 | {v} | R2
                            mR2 = R1 | new_L2
                            if not (mL2 & mR2) and _is_valid_biclique(mL2, mR2, all_edges_set):
                                ne = _cross_edges(mL2, mR2)
                                if ne == old_e:
                                    bicliques[i] = (mL2, mR2)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                    if improved:
                        break
                    for v in list(R2):
                        new_R2 = R2 - {v}
                        if not new_R2:
                            continue
                        if _is_valid_biclique(L2, new_R2, all_edges_set) and _is_valid_biclique(L1, R1 | {v}, all_edges_set):
                            mL = L1 | L2
                            mR = R1 | {v} | new_R2
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[i] = (mL, mR)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                        if _is_valid_biclique(L2, new_R2, all_edges_set) and _is_valid_biclique(L1, R1 | {v}, all_edges_set):
                            mL2 = L1 | R1 | {v}
                            mR2 = R2 | L2
                            if not (mL2 & mR2) and _is_valid_biclique(mL2, mR2, all_edges_set):
                                ne = _cross_edges(mL2, mR2)
                                if ne == old_e:
                                    bicliques[i] = (mL2, mR2)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Phase 4: Final absorption pass - try to absorb small bicliques into larger ones
    if time.time() < deadline:
        for _iter in range(6):
            if time.time() > deadline:
                break
            improved = False
            # Sort by size (smallest first) to try absorbing small into large
            order = sorted(range(len(bicliques)), key=lambda i: len(bicliques[i][0]) + len(bicliques[i][1]))
            for si in order:
                if time.time() > deadline:
                    break
                if si >= len(bicliques):
                    continue
                for ti in order:
                    if time.time() > deadline:
                        break
                    if ti == si or ti >= len(bicliques):
                        continue
                    if len(bicliques[si][0]) + len(bicliques[si][1]) > len(bicliques[ti][0]) + len(bicliques[ti][1]):
                        continue  # si should be smaller
                    Ls, Rs = bicliques[si]
                    Lt, Rt = bicliques[ti]
                    old_e = _cross_edges(Ls, Rs) | _cross_edges(Lt, Rt)
                    
                    # Try moving all of Ls to Lt
                    if not (Lt | Ls) & Rt and _is_valid_biclique(Lt | Ls, Rt, all_edges_set):
                        ne = _cross_edges(Lt | Ls, Rt)
                        if ne == old_e:
                            bicliques[ti] = (Lt | Ls, Rt)
                            bicliques.pop(si)
                            improved = True
                            break
                    # Try moving all of Ls to Rt
                    if not (Lt | Rs) & (Rt | Ls) and _is_valid_biclique(Lt | Rs, Rt | Ls, all_edges_set):
                        ne = _cross_edges(Lt | Rs, Rt | Ls)
                        if ne == old_e:
                            bicliques[ti] = (Lt | Rs, Rt | Ls)
                            bicliques.pop(si)
                            improved = True
                            break
                    # Try moving all of Rs to Lt
                    if not (Lt | Rs) & (Rt | Ls) and _is_valid_biclique(Lt | Rs, Rt | Ls, all_edges_set):
                        ne = _cross_edges(Lt | Rs, Rt | Ls)
                        if ne == old_e:
                            bicliques[ti] = (Lt | Rs, Rt | Ls)
                            bicliques.pop(si)
                            improved = True
                            break
                    # Try moving all of Rs to Rt
                    if not (Lt | Ls) & (Rt | Rs) and _is_valid_biclique(Lt | Ls, Rt | Rs, all_edges_set):
                        ne = _cross_edges(Lt | Ls, Rt | Rs)
                        if ne == old_e:
                            bicliques[ti] = (Lt | Ls, Rt | Rs)
                            bicliques.pop(si)
                            improved = True
                            break
                    # Try individual vertex moves from si to ti
                    for v in list(Ls):
                        new_Ls = Ls - {v}
                        if not new_Ls:
                            continue
                        if _is_valid_biclique(new_Ls, Rs, all_edges_set) and _is_valid_biclique(Lt | {v}, Rt, all_edges_set):
                            mL = new_Ls | Lt | {v}
                            mR = Rs | Rt
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[ti] = (mL, mR)
                                    bicliques.pop(si)
                                    improved = True
                                    break
                        if _is_valid_biclique(new_Ls, Rs, all_edges_set) and _is_valid_biclique(Lt, Rt | {v}, all_edges_set):
                            mL = new_Ls | Lt
                            mR = Rs | Rt | {v}
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[ti] = (mL, mR)
                                    bicliques.pop(si)
                                    improved = True
                                    break
                    if improved:
                        break
                    for v in list(Rs):
                        new_Rs = Rs - {v}
                        if not new_Rs:
                            continue
                        if _is_valid_biclique(Ls, new_Rs, all_edges_set) and _is_valid_biclique(Lt | {v}, Rt, all_edges_set):
                            mL = Ls | Lt | {v}
                            mR = new_Rs | Rt
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[ti] = (mL, mR)
                                    bicliques.pop(si)
                                    improved = True
                                    break
                        if _is_valid_biclique(Ls, new_Rs, all_edges_set) and _is_valid_biclique(Lt, Rt | {v}, all_edges_set):
                            mL = Ls | Lt
                            mR = new_Rs | Rt | {v}
                            if not (mL & mR) and _is_valid_biclique(mL, mR, all_edges_set):
                                ne = _cross_edges(mL, mR)
                                if ne == old_e:
                                    bicliques[ti] = (mL, mR)
                                    bicliques.pop(si)
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Convert to lists of ints
    result = []
    for L, R in bicliques:
        result.append([list(int(x) for x in sorted(L)), list(int(x) for x in sorted(R))])
    
    return result
# EVOLVE-BLOCK-END
