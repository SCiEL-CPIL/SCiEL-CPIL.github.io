"""B_3[1] set construction: maximize cardinality with unique unordered triple sums.

STRUCTURAL CHANGE vs. the seed:
  - Feasibility of adding a value is computed with vectorized NumPy (one
    candidate over the whole set at once) instead of nested Python loops.
  - Search is a constraint-driven backtracking (pick the candidate that
    conflicts with the fewest already-used sums, recurse, undo) plus a
    swap-based local search (2-opt moves) -- a different optimization family
    than the seed's one-shot greedy + weighted-random restarts.
"""
import time
import random
import numpy as np


def _triple_sums(S):
    """All unordered-with-replacement triple sums of S (values sorted)."""
    a = np.asarray(S, dtype=np.int64)
    n = a.size
    i, j, k = np.triu_indices(n, k=0)
    return a[i] + a[j] + a[k]


def _feasible_add(x, S_arr, sums_set):
    """True if adding x keeps every unordered triple sum unique."""
    n = S_arr.size
    if n == 0:
        return True
    # candidate sums involving x
    cand = np.empty(3 * n + 1, dtype=np.int64)
    cand[0] = 3 * x
    cand[1:n + 1] = 2 * x + S_arr          # (x, x, y)
    cand[n + 1:2 * n + 1] = x + 2 * S_arr  # (x, y, y)
    if n >= 2:
        ii, jj = np.triu_indices(n, k=1)
        cand[2 * n + 1:] = x + S_arr[ii] + S_arr[jj]  # (x, y, z), y<z
    return not any(s in sums_set for s in cand.tolist())


def _feasible_add_vec(x, S_arr, sums_arr):
    """Vectorized version (sums_arr = np.asarray of the current sum set)."""
    n = S_arr.size
    if n == 0:
        return True
    cand = np.empty(3 * n + 1, dtype=np.int64)
    cand[0] = 3 * x
    cand[1:n + 1] = 2 * x + S_arr
    cand[n + 1:2 * n + 1] = x + 2 * S_arr
    if n >= 2:
        ii, jj = np.triu_indices(n, k=1)
        cand[2 * n + 1:] = x + S_arr[ii] + S_arr[jj]
    return not np.any(np.isin(cand, sums_arr))


def _conflict_count(x, S_arr, sums_arr):
    """How many of x's candidate triple sums collide with existing sums."""
    n = S_arr.size
    if n == 0:
        return 0
    cand = np.empty(3 * n + 1, dtype=np.int64)
    cand[0] = 3 * x
    cand[1:n + 1] = 2 * x + S_arr
    cand[n + 1:2 * n + 1] = x + 2 * S_arr
    if n >= 2:
        ii, jj = np.triu_indices(n, k=1)
        cand[2 * n + 1:] = x + S_arr[ii] + S_arr[jj]
    return int(np.count_nonzero(np.isin(cand, sums_arr)))


def _backtrack_build(pool, deadline, rng, target_extra=8):
    """Constraint-driven backtracking: at each step pick the candidate with
    the fewest conflicts (tie -> smaller value), recurse, undo.  Returns a
    valid set (sorted).  Pool is a list of candidate values to draw from."""
    S = []
    sums = set()
    sums_arr = np.empty(0, dtype=np.int64)

    def _recurse(depth):
        nonlocal sums_arr
        if time.time() > deadline:
            return
        if len(S) >= len(pool) + target_extra:
            return
        # candidate values not yet in S
        in_S = set(S)
        cands = [v for v in pool if v not in in_S]
        if not cands:
            return
        # score each candidate by its conflict count with current sums
        S_arr = np.asarray(S, dtype=np.int64)
        best = None
        best_score = None
        for v in cands:
            c = _conflict_count(v, S_arr, sums_arr)
            # prefer low conflict, then small value (denser packing)
            score = (c, v)
            if best_score is None or score < best_score:
                best_score = score
                best = v
            if best_score[0] == 0 and best_score[1] == min(cands):
                break
        if best is None:
            return
        # add best
        S.append(best)
        S_arr = np.asarray(S, dtype=np.int64)
        new_sums = _triple_sums(S_arr)
        new_sums = new_sums[new_sums > 0]
        added = [s for s in new_sums.tolist() if s not in sums]
        # only the genuinely new sums (those involving best)
        # recompute: sums involving best
        n0 = len(S) - 1
        base = np.asarray(S[:n0], dtype=np.int64)
        if n0 == 0:
            inv = np.array([3 * best], dtype=np.int64)
        else:
            inv = np.empty(3 * n0 + 1, dtype=np.int64)
            inv[0] = 3 * best
            inv[1:n0 + 1] = 2 * best + base
            inv[n0 + 1:2 * n0 + 1] = best + 2 * base
            if n0 >= 2:
                ii, jj = np.triu_indices(n0, k=1)
                inv[2 * n0 + 1:] = best + base[ii] + base[jj]
        for s in inv.tolist():
            if s in sums:
                # conflict: undo and try next candidate
                S.pop()
                return
            sums.add(s)
        sums_arr = np.array(sorted(sums), dtype=np.int64)
        _recurse(depth + 1)
        # undo
        for s in inv.tolist():
            sums.discard(s)
        S.pop()
        sums_arr = np.array(sorted(sums), dtype=np.int64)

    _recurse(0)
    return sorted(S)


def _swap_local_search(S, deadline, rng, max_rounds=200):
    """2-opt swap search: try replacing one element with a better candidate.
    A swap is accepted if it keeps the set valid and (heuristic) does not
    reduce future flexibility.  Returns improved (or equal) sorted list."""
    S = list(S)
    in_S = set(S)
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        S_arr = np.asarray(S, dtype=np.int64)
        sums = _triple_sums(S_arr)
        sums_set = set(sums.tolist())
        # candidate values not in S
        cands = [v for v in range(1, 4097) if v not in in_S]
        if not cands:
            break
        improved = False
        # try each position; find a swap that is feasible
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            old = S[idx]
            # sums that involve old (must be removed)
            S_no_old = [S[t] for t in range(len(S)) if t != idx]
            S_no_old_arr = np.asarray(S_no_old, dtype=np.int64)
            # try candidates in a randomized order biased to small
            trial = cands[:]
            rng.shuffle(trial)
            trial.sort(key=lambda v: (v,))  # small first for packing
            for new in trial:
                # build candidate set
                cand_set = sorted(S_no_old + [new])
                cand_arr = np.asarray(cand_set, dtype=np.int64)
                new_sums = _triple_sums(cand_arr)
                new_set = set(new_sums.tolist())
                if len(new_set) == new_sums.size:  # all unique
                    S = cand_set
                    in_S = set(S)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return S


def _greedy_vectorized(pool, deadline):
    """Fast vectorized greedy: iterate pool, add if feasible."""
    S = []
    sums_set = set()
    sums_arr = np.empty(0, dtype=np.int64)
    for x in pool:
        if time.time() > deadline:
            break
        S_arr = np.asarray(S, dtype=np.int64)
        if _feasible_add(x, S_arr, sums_set):
            S.append(x)
            S_arr = np.asarray(S, dtype=np.int64)
            n0 = len(S) - 1
            base = np.asarray(S[:n0], dtype=np.int64)
            if n0 == 0:
                inv = np.array([3 * x], dtype=np.int64)
            else:
                inv = np.empty(3 * n0 + 1, dtype=np.int64)
                inv[0] = 3 * x
                inv[1:n0 + 1] = 2 * x + base
                inv[n0 + 1:2 * n0 + 1] = x + 2 * base
                if n0 >= 2:
                    ii, jj = np.triu_indices(n0, k=1)
                    inv[2 * n0 + 1:] = x + base[ii] + base[jj]
            for s in inv.tolist():
                sums_set.add(s)
            sums_arr = np.array(sorted(sums_set), dtype=np.int64)
    return sorted(S)


def construct():
    start = time.time()
    deadline = start + 80  # safety margin under the 90s kill
    rng = random.Random(12345)

    best = []

    # Phase 1: fast vectorized greedy in a few orders to get a solid seed set.
    for order in (list(range(1, 4097)),
                  list(range(4096, 0, -1)),
                  sorted(range(1, 4097), key=lambda v: (v % 7, v))):
        if time.time() > deadline:
            break
        S = _greedy_vectorized(order, deadline)
        if len(S) > len(best):
            best = S

    # Phase 2: constraint-driven backtracking from several candidate pools.
    pools = [
        list(range(1, 4097)),
        list(range(1, 2049)),
        list(range(1, 4097, 2)),
        list(range(1, 4097, 3)),
        sorted(range(1, 4097), key=lambda v: (v % 5, v)),
    ]
    for pool in pools:
        if time.time() > deadline:
            break
        S = _backtrack_build(pool, deadline, rng)
        if len(S) > len(best):
            best = S

    # Phase 3: swap-based local search to densify the best set.
    if time.time() < deadline:
        S = _swap_local_search(best, deadline, rng)
        if len(S) > len(best):
            best = S

    # Phase 4: a few more backtracking restarts with randomized pools,
    # then re-run swap search on the best.
    for _ in range(6):
        if time.time() > deadline:
            break
        pool = list(range(1, 4097))
        rng.shuffle(pool)
        S = _backtrack_build(pool, deadline, rng)
        if len(S) > len(best):
            best = S
        if time.time() < deadline - 5:
            S2 = _swap_local_search(S, deadline, rng)
            if len(S2) > len(best):
                best = S2

    # Final validation guard: ensure returned set is valid; if not, shrink.
    def _valid(S):
        if not S:
            return True
        a = np.asarray(S, dtype=np.int64)
        s = _triple_sums(a)
        return s.size == len(set(s.tolist()))

    if not _valid(best):
        # fall back to a safe greedy result
        best = _greedy_vectorized(list(range(1, 4097)), deadline)

    return sorted(set(best))


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    dt = time.time() - t0
    print(f"Time: {dt:.2f}s, Size: {len(res)}")
    a = np.asarray(res, dtype=np.int64)
    s = _triple_sums(a)
    print("Valid:", s.size == len(set(s.tolist())))
    print("Max value:", max(res) if res else 0)
