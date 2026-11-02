import time
import numpy as np
from itertools import combinations

D = 11
DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _generate_candidates():
    """All nonzero integer vectors with squared norm 4 in R^11 (the D_11 shell)."""
    vecs = []
    # Type A: single coordinate ±2  -> 2*D
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int32)
            v[i] = 2 * s
            vecs.append(v)
    # Type B: four coordinates ±1 -> C(D,4)*2^4
    for idxs in combinations(range(D), 4):
        for signs in np.ndindex(2, 2, 2, 2):
            v = np.zeros(D, dtype=np.int32)
            for idx, bit in zip(idxs, signs):
                v[idx] = 1 if bit == 0 else -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int32)


def _greedy_color_bound(compat, P, n):
    """Greedy coloring of the compat subgraph induced by P.
    Returns (color_of, color_classes); number of colors is an upper bound on
    the size of any independent set contained in P."""
    P_set = set(P)
    degrees = {}
    for v in P:
        degrees[v] = int(np.sum(compat[v, :]))  # proxy ordering key
    order = sorted(P, key=lambda x: -degrees[x])
    color_of = {}
    color_classes = []
    uncolored = list(order)
    while uncolored:
        c = len(color_classes)
        color_class = []
        remaining = []
        for v in uncolored:
            ok = True
            for u in color_class:
                if not compat[v, u]:
                    ok = False
                    break
            if ok:
                color_class.append(v)
                color_of[v] = c
            else:
                remaining.append(v)
        color_classes.append(color_class)
        uncolored = remaining
    return color_of, color_classes


def _bb_mcs(compat, n, deadline, best_size_ref):
    """Tomita BB-MCS for Maximum Independent Set on the compat graph."""
    best_size = best_size_ref[0]
    best_set_ref = [None]

    def _search(P, current_set):
        nonlocal best_size
        if time.time() > deadline:
            return
        if not P:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set_ref[0] = list(current_set)
                best_size_ref[0] = best_size
            return
        if len(current_set) + len(P) <= best_size:
            return
        color_of, color_classes = _greedy_color_bound(compat, P, n)
        num_colors = len(color_classes)
        for ci in range(num_colors - 1, -1, -1):
            if time.time() > deadline:
                return
            if len(current_set) + ci + 1 <= best_size:
                break
            color_class = color_classes[ci]
            for v in color_class:
                if v not in P:
                    continue
                new_set = current_set + [v]
                new_P = [u for u in P if u != v and compat[v, u]]
                _search(new_P, new_set)
                if time.time() > deadline:
                    return
                P.remove(v)

    P_initial = list(range(n))
    _search(P_initial, [])
    return best_set_ref[0]


def _local_search2opt(compat, selected, deadline):
    """2-opt swap: remove one selected vector, add any vector compatible with the rest."""
    selected = list(selected)
    n = compat.shape[0]
    in_set = np.zeros(n, dtype=bool)
    for s in selected:
        in_set[s] = True
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            removed = selected[i]
            rest = [x for x in selected if x != removed]
            # candidate set = vectors not in (rest) and not removed
            cand = [x for x in range(n) if not in_set[x] or x == removed]
            # need x != removed and x compatible with all in rest
            added = False
            for x in cand:
                if x == removed:
                    continue
                ok = True
                for r in rest:
                    if not compat[x, r]:
                        ok = False
                        break
                if ok:
                    selected.append(x)
                    in_set[x] = True
                    improved = True
                    added = True
                    break
            if added:
                continue
            # no addition possible: restore (removed stays out only if we can't re-add)
            # re-add removed to keep set valid/unchanged
            if removed not in selected:
                selected.append(removed)
    return selected


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    candidates = _generate_candidates()
    n = len(candidates)

    # compat[i,j] True iff vectors i and j can coexist: |dot| <= 2 (norm 4)
    dots = candidates @ candidates.T
    compat = (np.abs(dots) <= 2).astype(bool)
    np.fill_diagonal(compat, False)

    rng = np.random.default_rng(12345)
    best_size = 0
    best_set = []

    def _greedy(order):
        selected = []
        sel_mask = np.zeros(n, dtype=bool)
        for idx in order:
            if not selected:
                selected.append(idx)
                sel_mask[idx] = True
                continue
            if np.all(compat[idx, sel_mask]):
                selected.append(idx)
                sel_mask[idx] = True
        return selected

    # diverse orderings
    compat_count = compat.sum(axis=1)
    orderings = [np.argsort(-compat_count), np.argsort(compat_count), np.arange(n)]
    for _ in range(60):
        orderings.append(rng.permutation(n))

    for order in orderings:
        if time.time() > deadline:
            break
        s = _greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # 2-opt local search from the best greedy set
    if time.time() < deadline - 3.0:
        ls = _local_search2opt(compat, best_set, deadline - 3.0)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls

    # Tomita BB-MCS exact-improvement pass, seeded with current best
    best_size_ref = [best_size]
    if deadline - time.time() > 2.0:
        res = _bb_mcs(compat, n, deadline, best_size_ref)
        if res and len(res) > best_size:
            best_size = len(res)
            best_set = res

    if best_size == 0:
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    result = candidates[best_set].astype(np.int64)

    # Verify: min pairwise distance >= max norm
    n_res = len(result)
    max_norm_sq = int(np.max(np.sum(result ** 2, axis=1)))
    valid = True
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if np.dot(result[i] - result[j], result[i] - result[j]) < max_norm_sq:
                valid = False
                break
        if not valid:
            break

    if not valid:
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    return result
