"""Kissing number in dimension 11: max independent set on the conflict graph
of weight-<=4 integer vectors via GRASP (randomized min-degree construction
+ restricted-candidate-list 1-for-1 and 2-for-1 local search)."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def _build_pool():
    c = []
    # Type A: single 1  (squared norm 1)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            c.append(v)
    # Type B: single 2  (squared norm 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            c.append(v)
    # Type C: two 1s    (squared norm 2)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product((1, -1), repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            c.append(v)
    # Type D: four 1s   (squared norm 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            c.append(v)
    return np.array(c, dtype=np.int16)


def _greedy_construct(Conflict, N, deadline, rng):
    active = np.ones(N, dtype=bool)
    deg = Conflict.sum(axis=1).astype(np.int64)
    selected = []
    while active.any():
        if time.time() > deadline:
            break
        aidx = np.where(active)[0]
        ad = deg[aidx]
        mn = ad.min()
        cand = aidx[ad == mn]
        idx = int(cand[rng.integers(len(cand))])
        selected.append(idx)
        removed = Conflict[idx].astype(bool)
        removed[idx] = True
        removed = removed & active
        if removed.any():
            deg[active] -= Conflict[removed, :].sum(axis=0)[active]
        active[idx] = False
        active[removed] = False
    return selected


def _local_search(Conflict, N, selected, deadline, rng, max_rounds=12):
    sel = list(selected)
    selset = set(sel)
    improved = True
    rnd = 0
    while improved and rnd < max_rounds and time.time() < deadline:
        improved = False
        rnd += 1
        order = list(sel)
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline:
                return sel
            if i not in selset:
                continue
            rest = selset - {i}
            # 1-for-1: neighbor of i that is conflict-free with rest
            for j in np.where(Conflict[i])[0]:
                if j in selset:
                    continue
                if not np.any(Conflict[j, list(rest)]):
                    selset.discard(i)
                    selset.add(j)
                    sel = list(selset)
                    improved = True
                    break
            if improved:
                continue
            # 2-for-1: two mutually-compatible neighbors of i, both conflict-free with rest
            neigh = [int(j) for j in np.where(Conflict[i])[0] if j not in selset]
            if len(neigh) < 2:
                continue
            rest_arr = np.array(list(rest))
            if len(rest_arr) == 0:
                conflict_with_rest = np.zeros(len(neigh), dtype=bool)
            else:
                conflict_with_rest = Conflict[neigh, rest_arr].any(axis=1)
            ok = [neigh[k] for k in range(len(neigh)) if not conflict_with_rest[k]]
            if len(ok) < 2:
                continue
            for a in range(len(ok)):
                j1 = ok[a]
                for b in range(a + 1, len(ok)):
                    j2 = ok[b]
                    if not Conflict[j1, j2]:
                        selset.discard(i)
                        selset.add(j1)
                        selset.add(j2)
                        sel = list(selset)
                        improved = True
                        break
                if improved:
                    break
    return sel


def construct():
    start_time = time.time()
    deadline = start_time + 143  # safety margin under 150s

    candidates = _build_pool()
    N = len(candidates)

    M = (candidates @ candidates.T).astype(np.int32)
    norms_sq = np.sum(candidates.astype(np.int64) ** 2, axis=1)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * M
    Conflict = (dist_sq < 4).astype(np.int8)
    np.fill_diagonal(Conflict, 0)

    best = []
    best_size = 0

    def try_solution(sel):
        nonlocal best, best_size
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)

    # Phase 1: many randomized minimum-degree greedy constructions
    while time.time() < deadline:
        rng = np.random.default_rng()
        sel = _greedy_construct(Conflict, N, deadline, rng)
        try_solution(sel)
        if time.time() > deadline - 30:
            break

    # Phase 2: GRASP local-search refinement on fresh and improving solutions
    while time.time() < deadline - 5:
        rng = np.random.default_rng()
        if time.time() < deadline - 25:
            sel = _greedy_construct(Conflict, N, deadline, rng)
        else:
            sel = list(best)
        sel = _local_search(Conflict, N, sel, deadline, rng)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        elif time.time() > deadline - 10:
            break

    if not best:
        best = list(range(min(N, 22)))

    return candidates[best].astype(np.int64)
# EVOLVE-BLOCK-END
