"""Biclique partition: maximal biclique enumeration + frequency-aware greedy + enhanced local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
_EDGE_SET = set()
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGE_SET.add((min(_a, _b), max(_a, _b)))


def _star_partition():
    res = []
    for u in range(_N):
        nbrs = [v for v in _ADJ[u] if v > u]
        if nbrs:
            res.append(([u], list(nbrs)))
    return res


def _valid_partition(bics):
    covered = set()
    for L, R in bics:
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in _EDGE_SET:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == _EDGE_SET


def _biclique_edges(L, R):
    es = set()
    for u in L:
        for v in R:
            es.add((min(u, v), max(u, v)))
    return es


def _grow_biclique_maximal(L, R, remaining):
    """Grow biclique to be maximal within remaining edges."""
    Lset = set(L)
    Rset = set(R)
    changed = True
    while changed:
        changed = False
        # Try adding to L
        candidates = set()
        for v in Rset:
            candidates |= _ADJ[v]
        for u in candidates:
            if u not in Lset and u not in Rset:
                if all((min(u, v), max(u, v)) in remaining for v in Rset):
                    Lset.add(u)
                    changed = True
        # Try adding to R
        candidates = set()
        for u in Lset:
            candidates |= _ADJ[u]
        for v in candidates:
            if v not in Rset and v not in Lset:
                if all((min(u, v), max(u, v)) in remaining for u in Lset):
                    Rset.add(v)
                    changed = True
    return list(Lset), list(Rset)


def _enumerate_bicliques(deadline):
    """Enumerate a good set of bicliques by growing from each edge."""
    bicliques = []
    seen = set()
    edges = list(_EDGE_SET)
    
    for a, b in edges:
        if time.time() >= deadline:
            break
        # Grow from edge (a,b)
        L = {a}
        R = {v for v in _ADJ[a] if (min(a, v), max(a, v)) in _EDGE_SET}
        if b not in R:
            continue
        # Ensure b is in R and a is in L
        L, R = _grow_biclique_maximal(list(L), list(R), _EDGE_SET)
        # Normalize for dedup
        L_sorted = tuple(sorted(L))
        R_sorted = tuple(sorted(R))
        key = (L_sorted, R_sorted)
        if key not in seen:
            seen.add(key)
            bicliques.append((list(L), list(R)))
    
    # Also try growing from each vertex as left side
    for u in range(_N):
        if time.time() >= deadline:
            break
        R = [v for v in _ADJ[u] if (min(u, v), max(u, v)) in _EDGE_SET]
        if R:
            L, R = _grow_biclique_maximal([u], R, _EDGE_SET)
            L_sorted = tuple(sorted(L))
            R_sorted = tuple(sorted(R))
            key = (L_sorted, R_sorted)
            if key not in seen:
                seen.add(key)
                bicliques.append((list(L), list(R)))
    
    return bicliques


def _frequency_aware_greedy(bicliques, deadline):
    """Greedy set cover with frequency-aware scoring."""
    remaining = set(_EDGE_SET)
    bics = []
    
    # Precompute edge frequencies
    edge_freq = {e: 0 for e in _EDGE_SET}
    for L, R in bicliques:
        for e in _biclique_edges(L, R):
            edge_freq[e] += 1
    
    while remaining and time.time() < deadline:
        best_L = None
        best_R = None
        best_score = -1
        
        for L, R in bicliques:
            edges = _biclique_edges(L, R)
            # Only consider edges that are still remaining
            valid_edges = edges & remaining
            if not valid_edges:
                continue
            
            # Score: sum of inverse frequencies of covered edges
            # This prioritizes covering rare edges
            score = sum(1.0 / max(1, edge_freq[e]) for e in valid_edges)
            
            if score > best_score:
                best_score = score
                best_L = L
                best_R = R
        
        if best_L is None:
            # Fallback: pick any remaining edge
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
        
        bics.append((list(best_L), list(best_R)))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))
    
    # Add any leftover edges as single-edge bicliques
    for e in list(remaining):
        bics.append(([e[0]], [e[1]]))
    
    return bics


def _try_merge(bics, i, j):
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    L = list(set(L1) | set(L2))
    R = list(set(R1) | set(R2))
    if not L or not R:
        return None
    if set(L) & set(R):
        return None
    for u in L:
        for v in R:
            if (min(u, v), max(u, v)) not in _EDGE_SET:
                return None
    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    eu = _biclique_edges(L, R)
    if eu != (e1 | e2):
        return None
    new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
    new_bics.append((L, R))
    return new_bics


def _try_merge_three(bics, i, j, k):
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    L3, R3 = bics[k]
    L = list(set(L1) | set(L2) | set(L3))
    R = list(set(R1) | set(R2) | set(R3))
    if not L or not R:
        return None
    if set(L) & set(R):
        return None
    for u in L:
        for v in R:
            if (min(u, v), max(u, v)) not in _EDGE_SET:
                return None
    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    e3 = _biclique_edges(L3, R3)
    eu = _biclique_edges(L, R)
    if eu != (e1 | e2 | e3):
        return None
    new_bics = [b for idx, b in enumerate(bics) if idx not in (i, j, k)]
    new_bics.append((L, R))
    return new_bics


def _local_search(bics, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        # Try pairwise merges
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _try_merge(bics, i, j)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
        # Try triple merges
        if not improved and n >= 3:
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() >= deadline:
                        break
                    for k in range(j + 1, n):
                        new_bics = _try_merge_three(bics, i, j, k)
                        if new_bics is not None:
                            bics = new_bics
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return bics


def _reoptimize(bics, deadline):
    """Reoptimize by repacking all edges using the biclique pool."""
    all_edges = set()
    for L, R in bics:
        all_edges |= _biclique_edges(L, R)
    
    # Enumerate bicliques for these edges
    bicliques = _enumerate_bicliques(deadline - 2.0)
    
    # Filter to only bicliques that cover subsets of all_edges
    filtered = []
    for L, R in bicliques:
        edges = _biclique_edges(L, R)
        if edges <= all_edges:
            filtered.append((L, R))
    
    if not filtered:
        return bics
    
    new_bics = _frequency_aware_greedy(filtered, deadline - 1.0)
    new_bics = _local_search(new_bics, deadline)
    
    if len(new_bics) < len(bics):
        return new_bics
    return bics


def construct():
    deadline = time.time() + 85.0
    
    best_bics = None
    best_count = float('inf')
    
    # Enumerate bicliques once
    bicliques = _enumerate_bicliques(deadline - 30.0)
    
    # Try multiple greedy strategies
    strategies = []
    
    # Strategy 1: Frequency-aware greedy
    bics1 = _frequency_aware_greedy(bicliques, deadline - 20.0)
    bics1 = _local_search(bics1, deadline - 10.0)
    strategies.append(bics1)
    
    # Strategy 2: Standard greedy (largest first)
    remaining = set(_EDGE_SET)
    bics2 = []
    while remaining and time.time() < deadline - 10.0:
        best_L = None
        best_R = None
        best_count = 0
        for L, R in bicliques:
            edges = _biclique_edges(L, R) & remaining
            if len(edges) > best_count:
                best_count = len(edges)
                best_L = L
                best_R = R
        if best_L is None:
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
        bics2.append((list(best_L), list(best_R)))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))
    for e in list(remaining):
        bics2.append(([e[0]], [e[1]]))
    bics2 = _local_search(bics2, deadline - 10.0)
    strategies.append(bics2)
    
    # Evaluate all strategies
    for bics in strategies:
        if not _valid_partition(bics):
            continue
        if len(bics) < best_count:
            best_count = len(bics)
            best_bics = bics
    
    # Try reoptimization on the best
    if best_bics is not None and time.time() < deadline - 5.0:
        reopt = _reoptimize(best_bics, deadline)
        if _valid_partition(reopt) and len(reopt) < best_count:
            best_bics = reopt
            best_count = len(reopt)
    
    if best_bics is None:
        best_bics = _star_partition()
    
    result = [[list(map(int, L)), list(map(int, R))] for L, R in best_bics]
    return result
# EVOLVE-BLOCK-END
