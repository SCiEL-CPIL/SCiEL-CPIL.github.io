"""Rewritten program: greedy biclique extraction + local-search refinement."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_EDGES = P['edges']
_EDGE_SET = set(frozenset(e) for e in _EDGES)
_DEADLINE = 75.0  # safety margin under 90s wall-clock


def _norm_edge(u, v):
    return (u, v) if u < v else (v, u)


def _biclique_edges(L, R):
    out = []
    for u in L:
        for v in R:
            out.append(_norm_edge(u, v))
    return out


def _is_valid_biclique(L, R):
    if not L or not R:
        return False
    if set(L) & set(R):
        return False
    for u in L:
        for v in R:
            if _norm_edge(u, v) not in _EDGE_SET:
                return False
    return True


def _greedy_extract(remaining):
    """Extract bicliques greedily from a set of (u,v) edges."""
    remaining = set(remaining)
    bicliques = []
    while remaining:
        # pick a random seed edge
        seed = random.choice(list(remaining))
        u, v = seed
        L = {u}
        R = {v}
        # grow: keep only vertices consistent with all current cross edges
        while True:
            grew = False
            # try adding to L: w can join L if for every r in R, (w,r) is an edge
            for w in list(remaining):
                if w in L or w in R:
                    continue
                # w must be on the "u side": for all r in R, edge(w,r) in graph
                ok = all(_norm_edge(w, r) in _EDGE_SET for r in R)
                if ok:
                    L.add(w)
                    grew = True
            # try adding to R: w can join R if for every l in L, edge(l,w) is an edge
            for w in list(remaining):
                if w in L or w in R:
                    continue
                ok = all(_norm_edge(l, w) in _EDGE_SET for l in L)
                if ok:
                    R.add(w)
                    grew = True
            if not grew:
                break
        # collect all edges covered by this biclique
        cov = set()
        for u2 in L:
            for v2 in R:
                cov.add(_norm_edge(u2, v2))
        # remove covered edges from remaining
        remaining -= cov
        bicliques.append((list(L), list(R)))
    return bicliques


def _partition_edges(bicliques):
    """Return dict edge -> index, or None if invalid (overlap/gap)."""
    m = {}
    for i, (L, R) in enumerate(bicliques):
        for u in L:
            for v in R:
                e = _norm_edge(u, v)
                if e in m:
                    return None
                m[e] = i
    return m


def _try_merge(bicliques, i, j):
    """Try to merge biclique j into i. Returns new list or None."""
    L1, R1 = bicliques[i]
    L2, R2 = bicliques[j]
    # We need to partition the union of edges so that all cross pairs are edges.
    # Try: put all L2 into L1 (keep R1) and all R2 into R1, but only if valid.
    # Actually merging two bicliques is tricky; we try to form a new biclique
    # from a superset. Simplest valid merge: if L2 subset of R1 and R2 subset of L1
    # etc. For safety, try to build a new biclique covering exactly the union.
    # We attempt: new_L = L1, new_R = R1, then add vertices from biclique j that
    # fit. If we can't cover all j's edges, fail.
    # Simpler: check if union of the two edge sets is itself a biclique.
    union_edges = set()
    for u in L1:
        for v in R1:
            union_edges.add(_norm_edge(u, v))
    for u in L2:
        for v in R2:
            union_edges.add(_norm_edge(u, v))
    # find a biclique (A,B) whose cross set == union_edges
    # A must be subset of L1∪L2∪R1∪R2. Try candidate A = L1, B = R1, then expand.
    A = set(L1)
    B = set(R1)
    changed = True
    while changed:
        changed = False
        for w in set(L1) | set(L2) | set(R1) | set(R2):
            if w in A or w in B:
                continue
            if all(_norm_edge(w, b) in _EDGE_SET for b in B):
                A.add(w); changed = True
            elif all(_norm_edge(a, w) in _EDGE_SET for a in A):
                B.add(w); changed = True
    # check exactness
    cov = set()
    for a in A:
        for b in B:
            cov.add(_norm_edge(a, b))
    if cov != union_edges:
        return None
    new_list = [b for k, b in enumerate(bicliques) if k != i and k != j]
    new_list.append((list(A), list(B)))
    return new_list


def _try_split(bicliques, i):
    """Try to split biclique i into two smaller ones that together cover same edges.
    Returns new list or None."""
    L, R = bicliques[i]
    # Try to find a proper nonempty subset L' of L such that (L',R) is a biclique
    # and (L-L', R) is a biclique, and the union of their edges == original.
    # Since (L,R) is a biclique, any subset also forms a biclique with R.
    # So we can always split L into two nonempty parts. But that increases count!
    # We only want splits that enable later merges. Skip for now (returns None).
    return None


def _try_reassign(bicliques, i, j, u):
    """Move vertex u from side of biclique i to biclique j if it helps.
    This is complex; skip for simplicity."""
    return None


def construct():
    t0 = time.time()
    random.seed(12345)

    # Initial greedy partition
    best = _greedy_extract(_EDGE_SET)
    best_map = _partition_edges(best)
    best_count = len(best)

    # Local search: try merges
    improved = True
    while improved and (time.time() - t0) < _DEADLINE:
        improved = False
        n = len(best)
        for i in range(n):
            for j in range(i + 1, n):
                if (time.time() - t0) > _DEADLINE:
                    break
                new_list = _try_merge(best, i, j)
                if new_list is not None:
                    if _partition_edges(new_list) is not None:
                        best = new_list
                        best_count = len(best)
                        improved = True
                        break
            if (time.time() - t0) > _DEADLINE:
                break
            if improved:
                break

    # Final: ensure validity
    if _partition_edges(best) is None:
        # fallback to seed-style partition
        best = [[ [u], [v for a, v in _EDGES if a == u] ] for u in range(_N) if any(a == u for a, v in _EDGES)]

    # Convert to required format: list of [left_list, right_list]
    result = []
    for L, R in best:
        result.append([list(L), list(R)])
    return result
# EVOLVE-BLOCK-END
