"""Biclique partition via maximal biclique enumeration + set-cover greedy + local refinement."""
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _ek(u, v):
    return (u, v) if u < v else (v, u)


def _validate(bicliques, edges_set):
    if not bicliques and not edges_set:
        return True
    seen = set()
    for bi in bicliques:
        if len(bi) != 2:
            return False
        left, right = bi
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        if not left or not right:
            return False
        for u in left:
            if not isinstance(u, int) or isinstance(u, bool):
                return False
            for v in right:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
                if u == v:
                    return False
                e = _ek(u, v)
                if e not in edges_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edges_set


def _star_partition(n, edges_set):
    adj = [set() for _ in range(n)]
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    result = []
    for u in range(n):
        if adj[u]:
            result.append([sorted(adj[u]), [u]])
    return result


def _find_maximal_bicliques(n, edges_set):
    """Find all maximal bicliques in the graph using a systematic approach."""
    adj = [set() for _ in range(n)]
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    bicliques = []
    seen = set()
    
    # For each pair of vertices, try to build a biclique
    # A biclique [L, R] means all (u,v) for u in L, v in R are edges
    # and u != v for all u in L, v in R
    
    # Strategy: for each subset of left vertices (small), find max right
    # Better: enumerate all possible bicliques by their edge set
    
    # Use the fact that a biclique is determined by its two sides
    # For efficiency, try each vertex as a left seed and grow
    
    for u in range(n):
        # Try u alone as left
        candidates = set()
        for v in adj[u]:
            if v != u:
                candidates.add(v)
        if not candidates:
            continue
        
        # For each candidate right vertex, try to expand
        for v in candidates:
            if u == v:
                continue
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Expand L: add w such that w connected to all in R
                for w in range(n):
                    if w not in L and w != v:
                        ok = all(_ek(w, r) in edges_set for r in R)
                        if ok:
                            L.add(w)
                            changed = True
                # Expand R: add w such that w connected to all in L
                for w in range(n):
                    if w not in R and w not in L:
                        ok = all(_ek(l, w) in edges_set for l in L)
                        if ok:
                            R.add(w)
                            changed = True
            
            if L and R and L != R:
                key = (frozenset(L), frozenset(R))
                alt = (frozenset(R), frozenset(L))
                if key not in seen and alt not in seen:
                    # Check it's maximal: no vertex can be added to either side
                    is_maximal = True
                    for w in range(n):
                        if w not in L and w not in R:
                            if all(_ek(w, r) in edges_set for r in R) and all(_ek(l, w) in edges_set for l in L):
                                is_maximal = False
                                break
                    if is_maximal:
                        seen.add(key)
                        bicliques.append((sorted(L), sorted(R)))
    
    # Also try bicliques where left has multiple vertices
    # For pairs (u, v) with u < v, try {u,v} as left
    for u in range(n):
        for v in range(u + 1, n):
            if _ek(u, v) not in edges_set:
                continue
            L = {u, v}
            # Find common neighbors not in L
            common = adj[u] & adj[v]
            common -= L
            if not common:
                continue
            R = common
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w not in L and w not in R:
                        if all(_ek(w, r) in edges_set for r in R) and all(_ek(l, w) in edges_set for l in L):
                            # Could go to L or R
                            # Add to L
                            L.add(w)
                            changed = True
                            break
                for w in range(n):
                    if w not in L and w not in R:
                        if all(_ek(l, w) in edges_set for l in L) and all(_ek(w, r) in edges_set for r in R):
                            R.add(w)
                            changed = True
                            break
            
            if L and R and L != R:
                key = (frozenset(L), frozenset(R))
                alt = (frozenset(R), frozenset(L))
                if key not in seen and alt not in seen:
                    is_maximal = True
                    for w in range(n):
                        if w not in L and w not in R:
                            if all(_ek(w, r) in edges_set for r in R) and all(_ek(l, w) in edges_set for l in L):
                                is_maximal = False
                                break
                    if is_maximal:
                        seen.add(key)
                        bicliques.append((sorted(L), sorted(R)))
    
    # Deduplicate: keep only truly maximal ones
    # A biclique (L1,R1) is contained in (L2,R2) if L1⊆L2 and R1⊆R2
    filtered = []
    for i, (Li, Ri) in enumerate(bicliques):
        contained = False
        for j, (Lj, Rj) in enumerate(bicliques):
            if i == j:
                continue
            if set(Li) <= set(Lj) and set(Ri) <= set(Rj) and (set(Li) != set(Lj) or set(Ri) != set(Rj)):
                contained = True
                break
        if not contained:
            filtered.append((Li, Ri))
    
    return filtered


def _greedy_set_cover(candidates, edges_set, deadline):
    """Greedy set cover: repeatedly pick the biclique covering most uncovered edges."""
    remaining = set(edges_set)
    chosen = []
    
    # Precompute edge set for each candidate
    cand_edges = []
    for L, R in candidates:
        es = frozenset(_ek(u, v) for u in L for v in R)
        cand_edges.append(es)
    
    while remaining and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i in range(len(candidates)):
            if not cand_edges[i]:
                continue
            cnt = len(cand_edges[i] & remaining)
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        remaining -= cand_edges[best_idx]
        chosen.append(candidates[best_idx])
        # Remove edges from other candidates
        for i in range(len(candidates)):
            if i != best_idx:
                cand_edges[i] = cand_edges[i] - set(cand_edges[best_idx])
    
    # Handle leftover edges
    for e in remaining:
        chosen.append([list(e), list(e[1:])])
    
    return chosen


def _try_merge(bicliques, edges_set):
    """Try to merge pairs of bicliques if their union is still a valid biclique."""
    n = len(bicliques)
    changed = True
    while changed and n > 1:
        changed = False
        for i in range(len(bicliques)):
            if i >= len(bicliques):
                break
            for j in range(i + 1, len(bicliques)):
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                new_left = set(li) | set(lj)
                new_right = set(ri) | set(rj)
                # Check all cross pairs are edges and no self-pair
                valid = True
                for u in new_left:
                    if u in new_right:
                        valid = False
                        break
                    for v in new_right:
                        if _ek(u, v) not in edges_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check coverage: new must cover exactly the union of old coverages
                cov_i = set(_ek(u, v) for u in li for v in ri)
                cov_j = set(_ek(u, v) for u in lj for v in rj)
                cov_new = set(_ek(u, v) for u in new_left for v in new_right)
                if cov_new == cov_i | cov_j:
                    bicliques[i] = [sorted(new_left), sorted(new_right)]
                    del bicliques[j]
                    changed = True
                    break
            if changed:
                break
    return bicliques


def _local_refinement(bicliques, edges_set, deadline):
    """Try vertex moves between bicliques to reduce count."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bicliques)
        for i in range(n):
            if i >= n:
                break
            li, ri = bicliques[i]
            for j in range(n):
                if i == j or j >= n:
                    continue
                lj, rj = bicliques[j]
                # Try moving vertex u from li to rj (if it connects to all in rj and removing from li keeps i valid)
                for u in list(li):
                    if u in rj:
                        continue
                    # Check u connects to all in rj
                    if not all(_ek(u, v) in edges_set for v in rj):
                        continue
                    # Check removing u from li keeps biclique i valid (non-empty)
                    new_li = [x for x in li if x != u]
                    if not new_li:
                        continue
                    # Check no edge conflict: (u, v) for v in new_li shouldn't cause issues
                    # Actually we need: new biclique i = (new_li, ri) still valid
                    # And new biclique j = (lj, rj + [u]) still valid
                    # And no overlapping edges between new i and new j
                    new_ri_j = rj + [u]
                    # Check j still valid
                    ok_j = all(_ek(a, b) in edges_set for a in lj for b in new_ri_j) and u not in lj
                    if not ok_j:
                        continue
                    # Check no overlap
                    cov_i = set(_ek(a, b) for a in new_li for b in ri)
                    cov_j = set(_ek(a, b) for a in lj for b in new_ri_j)
                    if cov_i & cov_j:
                        continue
                    # Check total coverage is preserved
                    old_cov_i = set(_ek(a, b) for a in li for b in ri)
                    old_cov_j = set(_ek(a, b) for a in lj for b in rj)
                    if cov_i | cov_j == old_cov_i | old_cov_j:
                        bicliques[i] = [new_li, ri]
                        bicliques[j] = [lj, new_ri_j]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        # Try absorbing: merge biclique i into j if possible
        if not improved:
            for i in range(n):
                if i >= n:
                    break
                for j in range(n):
                    if i == j or j >= n:
                        continue
                    li, ri = bicliques[i]
                    lj, rj = bicliques[j]
                    # Can all edges of i be covered by extending j?
                    # Try: new j = (lj ∪ li, rj ∪ ri) if valid
                    new_left = set(lj) | set(li)
                    new_right = set(rj) | set(ri)
                    valid = True
                    for u in new_left:
                        if u in new_right:
                            valid = False
                            break
                        for v in new_right:
                            if _ek(u, v) not in edges_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        cov_new = set(_ek(u, v) for u in new_left for v in new_right)
                        cov_j = set(_ek(u, v) for u in lj for v in rj)
                        cov_i = set(_ek(u, v) for u in li for v in ri)
                        if cov_new == cov_j | cov_i:
                            bicliques[j] = [sorted(new_left), sorted(new_right)]
                            del bicliques[i]
                            improved = True
                            break
                if improved:
                    break
    return bicliques


def construct():
    start = time.time()
    deadline = start + 80  # 10s safety margin
    
    n = P['n']
    edges_set = set(_ek(u, v) for u, v in P['edges'])
    
    # Fallback
    fallback = _star_partition(n, edges_set)
    best = fallback
    best_count = len(fallback)
    
    # Find maximal bicliques
    candidates = _find_maximal_bicliques(n, edges_set)
    
    # Add single-edge bicliques for safety
    for e in edges_set:
        c = ([e[0]], [e[1]])
        if c not in candidates:
            candidates.append(c)
    
    # Greedy set cover with multiple random orderings
    random.seed(12345)
    for attempt in range(8):
        if time.time() > deadline - 5:
            break
        shuffled = candidates[:]
        if attempt > 0:
            random.shuffle(shuffled)
        result = _greedy_set_cover(shuffled, edges_set, deadline - 5)
        
        # Merge pass
        result = _try_merge(result, edges_set)
        
        # Local refinement
        if time.time() < deadline - 10:
            result = _local_refinement(result, edges_set, deadline - 10)
        
        if _validate(result, edges_set) and len(result) < best_count:
            best = result
            best_count = len(result)
    
    # Final merge and refinement
    best = _try_merge(best, edges_set)
    if time.time() < deadline - 2:
        best = _local_refinement(best, edges_set, deadline - 2)
    
    if not _validate(best, edges_set):
        best = fallback
    
    # Ensure correct types
    out = []
    for left, right in best:
        out.append([[int(u) for u in left], [int(v) for v in right]])
    
    return out
