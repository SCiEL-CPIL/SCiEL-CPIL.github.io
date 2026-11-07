"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def construct():
    start = time.time()
    deadline = start + 85.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    all_edges_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        all_edges_set.add(_edge_key(u, v))
    
    def _cross_edges(L, R):
        s = set()
        for l in L:
            for r in R:
                s.add(_edge_key(l, r))
        return s
    
    def _is_valid_biclique(L, R, edge_set):
        if L & R:
            return False
        for l in L:
            for r in R:
                if _edge_key(l, r) not in edge_set:
                    return False
        return True
    
    def _expand_biclique(u, v, rem_set):
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x in L or x in R:
                    continue
                if all(_edge_key(x, r) in rem_set for r in R):
                    L.add(x)
                    changed = True
            for y in range(n):
                if y in L or y in R:
                    continue
                if all(_edge_key(l, y) in rem_set for l in L):
                    R.add(y)
                    changed = True
        return L, R
    
    def _greedy_partition(remaining, strategy='edge_ratio'):
        """Greedy set-cover with different scoring strategies."""
        bicliques = []
        rem = set(remaining)
        
        while rem and time.time() < deadline:
            best_biclique = None
            best_covered = set()
            best_score = -1
            
            rem_list = list(rem)
            max_seeds = min(len(rem_list), 100) if len(rem_list) > 50 else len(rem_list)
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
            
            # Compute degrees in remaining subgraph
            deg = {v: 0 for v in range(n)}
            for u, v in rem:
                deg[u] += 1
                deg[v] += 1
            
            for u, v in seeds:
                L, R = _expand_biclique(u, v, rem)
                
                if L & R:
                    continue
                
                covered = set()
                for l in L:
                    for r in R:
                        k = _edge_key(l, r)
                        if k in rem:
                            covered.add(k)
                
                if not covered:
                    continue
                
                if strategy == 'edge_ratio':
                    score = len(covered) / max(1, len(L) + len(R))
                elif strategy == 'max_edges':
                    score = len(covered)
                elif strategy == 'compact':
                    score = len(covered) / max(1, (len(L) + len(R)) ** 1.2)
                else:
                    score = len(covered)
                
                if score > best_score:
                    best_score = score
                    best_biclique = (L, R)
                    best_covered = covered
            
            if best_biclique is None or not best_covered:
                k = next(iter(rem))
                best_biclique = ({k[0]}, {k[1]})
                best_covered = {k}
            
            bicliques.append(best_biclique)
            rem -= best_covered
        
        if rem:
            for k in list(rem):
                bicliques.append(({k[0]}, {k[1]}))
        
        return bicliques
    
    def _try_merge_pairs(bicliques, deadline):
        """Try to merge pairs of bicliques."""
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
                    
                    # Try all 4 merge combinations
                    for combo in [(L1 | L2, R1 | R2), (L1 | R2, R1 | L2), (R1 | L2, L1 | R2), (R1 | R2, L1 | L2)]:
                        new_L, new_R = combo
                        if not new_L or not new_R:
                            continue
                        if new_L & new_R:
                            continue
                        
                        if _is_valid_biclique(new_L, new_R, all_edges_set):
                            new_e = _cross_edges(new_L, new_R)
                            old_e = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                            if new_e == old_e:
                                bicliques[i] = (new_L, new_R)
                                bicliques.pop(j)
                                changed = True
                                break
                    
                    if changed:
                        break
                if changed:
                    break
        return bicliques
    
    def _local_search(bicliques, deadline):
        """Local search: move vertices between bicliques to enable merges."""
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
                    
                    # Try moving vertices from L1 to L2
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        new_L2 = L2 | {v}
                        if not new_L1 or not new_L2:
                            continue
                        if _is_valid_biclique(new_L1, R1, all_edges_set) and _is_valid_biclique(new_L2, R2, all_edges_set):
                            for combo in [(new_L1 | new_L2, R1 | R2), (new_L1 | R2, R1 | new_L2)]:
                                mL, mR = combo
                                if not mL or not mR or (mL & mR):
                                    continue
                                if _is_valid_biclique(mL, mR, all_edges_set):
                                    old_e = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                                    new_e = _cross_edges(mL, mR)
                                    if new_e == old_e:
                                        bicliques[i] = (mL, mR)
                                        bicliques.pop(j)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                    
                    # Try moving vertices from R1 to R2
                    for v in list(R1):
                        new_R1 = R1 - {v}
                        new_R2 = R2 | {v}
                        if not new_R1 or not new_R2:
                            continue
                        if _is_valid_biclique(L1, new_R1, all_edges_set) and _is_valid_biclique(L2, new_R2, all_edges_set):
                            for combo in [(L1 | L2, new_R1 | new_R2), (L1 | new_R2, new_R1 | L2)]:
                                mL, mR = combo
                                if not mL or not mR or (mL & mR):
                                    continue
                                if _is_valid_biclique(mL, mR, all_edges_set):
                                    old_e = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                                    new_e = _cross_edges(mL, mR)
                                    if new_e == old_e:
                                        bicliques[i] = (mL, mR)
                                        bicliques.pop(j)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                    
                    # Try moving vertices from L1 to R2 (side swap)
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        new_R2 = R2 | {v}
                        if not new_L1 or not new_R2:
                            continue
                        if _is_valid_biclique(new_L1, R1, all_edges_set) and _is_valid_biclique(L2, new_R2, all_edges_set):
                            for combo in [(new_L1 | L2, R1 | new_R2), (new_L1 | new_R2, R1 | L2)]:
                                mL, mR = combo
                                if not mL or not mR or (mL & mR):
                                    continue
                                if _is_valid_biclique(mL, mR, all_edges_set):
                                    old_e = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                                    new_e = _cross_edges(mL, mR)
                                    if new_e == old_e:
                                        bicliques[i] = (mL, mR)
                                        bicliques.pop(j)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                    
                    # Try moving vertices from R1 to L2 (side swap)
                    for v in list(R1):
                        new_R1 = R1 - {v}
                        new_L2 = L2 | {v}
                        if not new_R1 or not new_L2:
                            continue
                        if _is_valid_biclique(L1, new_R1, all_edges_set) and _is_valid_biclique(new_L2, R2, all_edges_set):
                            for combo in [(L1 | new_L2, new_R1 | R2), (L1 | R2, new_R1 | new_L2)]:
                                mL, mR = combo
                                if not mL or not mR or (mL & mR):
                                    continue
                                if _is_valid_biclique(mL, mR, all_edges_set):
                                    old_e = _cross_edges(L1, R1) | _cross_edges(L2, R2)
                                    new_e = _cross_edges(mL, mR)
                                    if new_e == old_e:
                                        bicliques[i] = (mL, mR)
                                        bicliques.pop(j)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return bicliques
    
    def _absorb_small(bicliques, deadline):
        """Try to absorb small bicliques into larger ones."""
        for _iter in range(6):
            if time.time() > deadline:
                break
            improved = False
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
                        continue
                    Ls, Rs = bicliques[si]
                    Lt, Rt = bicliques[ti]
                    
                    # Try various ways to absorb si into ti
                    for combo in [(Lt | Ls, Rt | Rs), (Lt | Rs, Rt | Ls), (Lt, Rt | Ls | Rs), (Lt | Ls | Rs, Rt)]:
                        new_L, new_R = combo
                        if not new_L or not new_R or (new_L & new_R):
                            continue
                        if _is_valid_biclique(new_L, new_R, all_edges_set):
                            old_e = _cross_edges(Ls, Rs) | _cross_edges(Lt, Rt)
                            new_e = _cross_edges(new_L, new_R)
                            if new_e == old_e:
                                bicliques[ti] = (new_L, new_R)
                                bicliques.pop(si)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return bicliques
    
    def _validate(bicliques):
        """Final validation."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            if L & R:
                return False
            for u in L:
                for v in R:
                    k = _edge_key(u, v)
                    if k not in all_edges_set:
                        return False
                    if k in covered:
                        return False
                    covered.add(k)
        return covered == all_edges_set
    
    # Generate multiple candidate partitions with different strategies
    candidates = []
    
    # Strategy 1: Edge ratio
    if time.time() < deadline:
        c1 = _greedy_partition(all_edges_set, 'edge_ratio')
        if _validate(c1):
            candidates.append(c1)
    
    # Strategy 2: Max edges
    if time.time() < deadline:
        c2 = _greedy_partition(all_edges_set, 'max_edges')
        if _validate(c2):
            candidates.append(c2)
    
    # Strategy 3: Compact
    if time.time() < deadline:
        c3 = _greedy_partition(all_edges_set, 'compact')
        if _validate(c3):
            candidates.append(c3)
    
    # Apply refinement to each candidate
    best = None
    best_count = float('inf')
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        # Merge pairs
        refined = _try_merge_pairs(list(cand), deadline - 2)
        
        # Local search
        refined = _local_search(refined, deadline - 1)
        
        # Absorb small
        refined = _absorb_small(refined, deadline)
        
        # Final merge pass
        refined = _try_merge_pairs(refined, deadline)
        
        if _validate(refined):
            if len(refined) < best_count:
                best_count = len(refined)
                best = refined
    
    # If no valid candidate, fall back to simple partition
    if best is None:
        best = _greedy_partition(all_edges_set, 'edge_ratio')
        best = _try_merge_pairs(best, deadline)
        best = _local_search(best, deadline)
        best = _absorb_small(best, deadline)
    
    # Convert to lists of ints
    result = []
    for L, R in best:
        result.append([list(int(x) for x in sorted(L)), list(int(x) for x in sorted(R))])
    
    return result
# EVOLVE-BLOCK-END
