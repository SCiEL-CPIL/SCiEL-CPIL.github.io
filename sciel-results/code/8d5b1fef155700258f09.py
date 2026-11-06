import time
import numpy as np

MAX_N = 1000
LIMIT = 2 * MAX_N  # 2000


def _feasible(rep, s):
    """True if all sums in s have rep < 2."""
    return int(np.max(rep[s])) < 2 if s.size else True


def _add(rep, s):
    rep[s] += 1


def _remove(rep, s):
    rep[s] -= 1


def _sums_with(x, arr):
    """All sums created by x with each element of arr, plus 2x."""
    if arr.size:
        return np.concatenate([arr + x, [2 * x]])
    return np.array([2 * x], dtype=np.int64)


def _tight(rep):
    return int(np.count_nonzero(rep == 2))


def _greedy(rng, chosen, rep, arr, deadline):
    """Randomized greedy add, maintaining arr incrementally."""
    chosen_set = set(chosen)
    while time.time() < deadline:
        # remaining candidates
        rem = [x for x in range(1, MAX_N + 1) if x not in chosen_set]
        if not rem:
            break
        # sample candidates to keep it fast
        if len(rem) > 300:
            sample = rng.choice(rem, size=300, replace=False)
        else:
            sample = np.array(rem)
        best = None
        best_score = None
        for x in sample:
            s = _sums_with(int(x), arr)
            if not _feasible(rep, s):
                continue
            score = int(np.count_nonzero(rep[s] == 1))
            if best_score is None or score < best_score:
                best_score = score
                best = int(x)
        if best is None:
            break
        s = _sums_with(best, arr)
        _add(rep, s)
        chosen.append(best)
        chosen_set.add(best)
        arr = np.append(arr, best)
    return chosen, rep, arr


def _local_search(rng, chosen, rep, arr, deadline):
    """Swap moves to reduce tight count."""
    chosen_set = set(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        outside = [x for x in range(1, MAX_N + 1) if x not in chosen_set]
        if not outside:
            break
        cur_tight = _tight(rep)
        n_trials = 500
        for _ in range(n_trials):
            if time.time() > deadline:
                break
            b = int(rng.choice(outside))
            a = int(rng.choice(arr))
            if b == a:
                continue
            idx = int(np.where(arr == a)[0][0])
            arr_no_a = np.concatenate([arr[:idx], arr[idx + 1:]])
            s_b = _sums_with(b, arr_no_a)
            if not _feasible(rep, s_b):
                continue
            s_a = _sums_with(a, arr_no_a)
            rep2 = rep.copy()
            _remove(rep2, s_a)
            _add(rep2, s_b)
            new_tight = _tight(rep2)
            if new_tight < cur_tight:
                rep = rep2
                chosen.remove(a)
                chosen_set.discard(a)
                chosen.append(b)
                chosen_set.add(b)
                arr = np.append(arr_no_a, b)
                cur_tight = new_tight
                improved = True
                break
    return chosen, rep, arr


def construct():
    deadline = time.time() + 110.0
    best = []
    best_size = 0
    rng = np.random.default_rng(0)

    while time.time() < deadline - 15:
        rep = np.zeros(LIMIT + 1, dtype=np.int32)
        chosen = []
        arr = np.array([], dtype=np.int64)
        # random init
        init_pool = list(range(1, MAX_N + 1))
        rng.shuffle(init_pool)
        init_count = int(rng.integers(0, 6))
        for x in init_pool[:init_count]:
            s = _sums_with(x, arr)
            if _feasible(rep, s):
                _add(rep, s)
                chosen.append(x)
                arr = np.append(arr, x)

        chosen, rep, arr = _greedy(rng, chosen, rep, arr, min(deadline - 8, time.time() + 12))
        if time.time() < deadline - 8:
            chosen, rep, arr = _local_search(rng, chosen, rep, arr, deadline - 4)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best = list(chosen)

    if not best:
        rep = np.zeros(LIMIT + 1, dtype=np.int32)
        chosen = []
        arr = np.array([], dtype=np.int64)
        for x in range(1, MAX_N + 1):
            s = _sums_with(x, arr)
            if _feasible(rep, s):
                _add(rep, s)
                chosen.append(x)
                arr = np.append(arr, x)
        best = chosen

    return list(best)
