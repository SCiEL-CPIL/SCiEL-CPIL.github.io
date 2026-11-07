"""Biclique partition via set-cover (randomized greedy + local search)."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_EDGE_SET = frozenset(frozenset(e) for e in P['edges'])
_ADJ = [set() for _ in range(P['n'])]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)

def _biclique_edges(L, R):
    """Set of frozenset edges covered by this biclique."""
    s = set()
    for u in L:
        for v in R:
            s.add(frozenset((u, v)))
    return s

def _valid_partition(bics):
    """Return (is_valid, covered_set, count) for a list of bicliques."""
    covered = set()
    for L, R in bics:
        if not L or not R:
            return False, covered, len(bics)
        if any(v in L for v in R):
            return False, covered, len(bics)
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e not in _EDGE_SET:
                    return False, covered, len(bics)
                if e in covered:
                    return False, covered, len(bics)
                covered.add(e)
    return covered == set(_EDGE_SET), covered, len(bics)

def _seed_star_partition():
    """Star partition: one biclique per vertex with its neighbors."""
    res = []
    for u in range(P['n']):
        nbrs = [v for v in _ADJ[u] if v > u]
        if nbrs:
            res.append(([u], nbrs))
    return res

def _grow_biclique(seed_edge, remaining):
    """Grow a biclique from a seed edge within the remaining edge set."""
    a, b = tuple(seed_edge)
    L = {a}
    R = {b}
    changed = True
    while changed:
        changed = False
        # Grow R
        for v in _ADJ[a]:
            if v in R or v in L:
                continue
            if all(frozenset((u, v)) in remaining for u in L):
                R.add(v)
                changed = True
        # Grow L
        for u in _ADJ[b]:
            if u in L or u in R:
                continue
            if all(frozenset((u, v)) in remaining for v in R):
                L.add(u)
                changed = True
    return list(L), list(R)

def _greedy_set_cover(remaining, deadline):
    """Randomized greedy set cover: pick biclique covering most remaining edges."""
    bics = []
    rem = set(remaining)
    while rem:
        if time.time() >= deadline:
            break
        # Try several random seed edges, pick the one giving largest coverage
        edges_list = list(rem)
        random.shuffle(edges_list)
        best_L, best_R, best_cov = None, None, -1
        for seed in edges_list[:20]:
            L, R = _grow_biclique(seed, rem)
            cov = _biclique_edges(L, R)
            n_covered = len(cov & rem)
            if n_covered > best_cov:
                best_cov = n_covered
                best_L, best_R = L, R
        if best_L is None:
            a, b = tuple(next(iter(rem)))
            best_L, best_R = [a], [b]
        bics.append((best_L, best_R))
        for u in best_L:
            for v in best_R:
                rem.discard(frozenset((u, v)))
    return bics

def _try_merge_pair(bics, i, j):
    """Try to merge bicliques i and j into one valid biclique covering exactly their union of edges."""
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    target = e1 | e2
    
    candidates = []
    # Union of both sides
    Lu = list(set(L1) | set(L2))
    Ru = list(set(R1) | set(R2))
    if not (set(Lu) & set(Ru)):
        candidates.append((Lu, Ru))
    # L union, R1
    L = list(set(L1) | set(L2))
    R = list(R1)
    if not (set(L) & set(R)):
        candidates.append((L, R))
    # L1, R union
    L = list(L1)
    R = list(set(R1) | set(R2))
    if not (set(L) & set(R)):
        candidates.append((L, R))
    # L2, R union
    L = list(L2)
    R = list(set(R1) | set(R2))
    if not (set(L) & set(R)):
        candidates.append((L, R))
    # L union, R2
    L = list(set(L1) | set(L2))
    R = list(R2)
    if not (set(L) & set(R)):
        candidates.append((L, R))
    # L1, R2 (if disjoint)
    if not (set(L1) & set(R2)):
        candidates.append((list(L1), list(R2)))
    # L2, R1 (if disjoint)
    if not (set(L2) & set(R1)):
        candidates.append((list(L2), list(R1)))
    
    for L, R in candidates:
        if not L or not R:
            continue
        # Check all cross pairs are edges
        ok = True
        for u in L:
            for v in R:
                if frozenset((u, v)) not in _EDGE_SET:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        eu = _biclique_edges(L, R)
        if eu == target:
            new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
            new_bics.append((L, R))
            return new_bics
    return None

def _try_split_replace(bics, i, j, deadline):
    """Try replacing bicliques i and j with a different 2-biclique decomposition of their combined edges.
    Returns new_bics if it reduces count (impossible for 2->2) or is equivalent but enables further merges.
    Actually for this we just try: can the combined edges be covered by 1 biclique? That's _try_merge_pair.
    So this function is a no-op placeholder for structural clarity."""
    return None

def _local_search_setcover(bics, deadline):
    """Local search: repeatedly try pairwise merges until fixed point or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _try_merge_pair(bics, i, j)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics

def _local_search_swap(bics, deadline):
    """2-opt style: try removing one biclique and re-covering its edges with other bicliques' extensions.
    For now, this is just another pass of merge attempts in random order."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        order = list(range(n))
        random.shuffle(order)
        for i_idx in order:
            if time.time() >= deadline:
                break
            for j_idx in order:
                if j_idx <= i_idx:
                    continue
                new_bics = _try_merge_pair(bics, i_idx, j_idx)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics

def _final_partition(bics):
    """Convert to required list-of-lists-of-int form."""
    return [[list(map(int, L)), list(map(int, R))] for L, R in bics]

def construct():
    deadline = time.time() + 80  # safety margin under 90s kill
    best_bics = None
    best_count = float('inf')
    
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() >= deadline - 15:
            break
        random.seed()
        bics = _greedy_set_cover(set(_EDGE_SET), deadline - 20)
        # Validate greedy result
        ok, covered, cnt = _valid_partition(bics)
        if not ok:
            # Fallback: star partition
            bics = _seed_star_partition()
        # Local search: merge pairs
        bics = _local_search_setcover(bics, deadline - 10)
        # Second pass with random order
        bics = _local_search_swap(bics, deadline - 5)
        # Final validation
        ok, covered, cnt = _valid_partition(bics)
        if ok and cnt < best_count:
            best_count = cnt
            best_bics = bics
    
    if best_bics is None:
        best_bics = _seed_star_partition()
    
    return _final_partition(best_bics)
# EVOLVE-BLOCK-END
