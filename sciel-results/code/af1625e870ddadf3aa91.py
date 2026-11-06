import time
import random
import numpy as np

_START = None
_DEADLINE = None
_LIMIT = 130.0  # safety margin under the 150s kill (task says ~120s)


def _now():
    return time.perf_counter()


def _diffs_sorted(arr):
    # arr is sorted; return all positive pairwise differences
    n = len(arr)
    out = np.empty(n * (n - 1) // 2, dtype=np.int64)
    k = 0
    for i in range(n):
        out[k:k + (n - 1 - i)] = arr[i + 1:] - arr[i]
        k += (n - 1 - i)
    return out


def _is_valid(arr):
    # arr sorted, must contain 0, all positive differences distinct, cover 1..30
    d = _diffs_sorted(arr)
    if d.size != len(set(d.tolist())):
        return False
    if len(set(d.tolist())) != d.size:
        return False
    # cover 1..30
    u = np.unique(d)
    for v in range(1, 31):
        if not np.any(u == v):
            return False
    return True


def _diameter(arr):
    return int(arr[-1] - arr[0])


def _score(arr):
    # lower is better
    if not _is_valid(arr):
        return float('inf')
    return _diameter(arr)


def _greedy_seed():
    # fast constructive seed: realize each missing difference with a new element
    a = [0]
    for d in range(1, 31):
        cur = sorted(a)
        ds = set(_diffs_sorted(np.array(cur, dtype=np.int64)).tolist())
        if d in ds:
            continue
        placed = False
        # try adding a single element x such that x-a or a-x == d for some a
        cands = []
        for v in cur:
            for s in (d, -d):
                x = v + s
                if x >= 0 and x not in set(cur):
                    cands.append(x)
        random.shuffle(cands)
        for x in cands:
            trial = np.array(sorted(cur + [x]), dtype=np.int64)
            if _is_valid(trial):
                a = trial.tolist()
                placed = True
                break
        if not placed:
            # fallback: add a pair far out
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = np.array(sorted(a + [y, y + d]), dtype=np.int64)
                if _is_valid(trial):
                    a = trial.tolist()
                    break
                y += 1
    return np.array(sorted(a), dtype=np.int64)


def _move_valid(base, idx, new):
    # base is sorted without idx; test inserting new at position idx
    n = len(base)
    # build full array
    if idx == 0:
        full = np.concatenate(([new], base))
    elif idx == n:
        full = np.concatenate((base, [new]))
    else:
        full = np.concatenate((base[:idx], [new], base[idx:]))
    if new < full[0] or new > full[-1]:
        return False
    # must be distinct and in range
    if full[idx] != new:
        return False
    # check distinctness locally: differences involving new must be unique and not collide
    # full differences involving index idx:
    # with elements before: new - full[j] for j < idx
    # with elements after: full[j] - new for j > idx
    before = new - full[:idx]
    after = full[idx + 1:] - new
    involved = np.concatenate((before, after))
    if len(set(involved.tolist())) != involved.size:
        return False
    # must not collide with any existing difference not involving idx
    # existing differences (full without idx) are base's differences
    base_d = _diffs_sorted(base)
    if len(np.intersect1d(involved, base_d)) > 0:
        return False
    # also base itself must be valid (we maintain validity of base)
    return True


def _local_search(arr, budget):
    # randomized hill-climbing on diameter
    arr = arr.copy()
    best = _diameter(arr)
    n = len(arr)
    rng = np.random.default_rng(12345)
    while _now() < budget:
        # pick a random element to perturb
        idx = int(rng.integers(0, n))
        # choose a target value near current
        cur = arr[idx]
        span = best + 5
        # propose new value
        new = int(rng.integers(0, best + 10))
        if new == cur:
            continue
        base = np.delete(arr, idx)
        if _move_valid(base, idx, new):
            trial = np.concatenate((base[:idx], [new], base[idx:]))
            trial = np.sort(trial)
            if _is_valid(trial):
                d = _diameter(trial)
                if d < best:
                    best = d
                    arr = trial
    return arr, best


def _shrink(arr, budget):
    # try to remove elements while staying valid (fewer elements -> smaller diameter possible)
    arr = arr.copy()
    best = _diameter(arr)
    rng = np.random.default_rng(999)
    while _now() < budget and len(arr) > 8:
        # try removing a random element (not 0)
        idx = int(rng.integers(1, len(arr)))
        trial = np.delete(arr, idx)
        if _is_valid(trial):
            d = _diameter(trial)
            if d <= best:
                arr = trial
                best = d
    return arr, best


def construct():
    global _START, _DEADLINE
    if _START is None:
        _START = _now()
        _DEADLINE = _START + _LIMIT
    start = _now()
    deadline = _START + _LIMIT

    # initial seed
    seed = _greedy_seed()
    best_arr = seed.copy()
    best_d = _diameter(seed)

    # local search with a chunked budget, checking deadline each chunk
    while _now() < deadline - 1.0:
        chunk = min(5.0, deadline - 1.0 - _now())
        if chunk <= 0:
            break
        arr, d = _local_search(best_arr, _now() + chunk)
        if d < best_d:
            best_d = d
            best_arr = arr
        # occasional shrink
        if _now() < deadline - 1.0:
            arr2, d2 = _shrink(best_arr, min(_now() + 1.0, deadline - 1.0))
            if d2 < best_d:
                best_d = d2
                best_arr = arr2
        # restart occasionally from a fresh seed to escape local minima
        if _now() < deadline - 3.0 and random.random() < 0.2:
            seed2 = _greedy_seed()
            if _diameter(seed2) < best_d:
                best_d = _diameter(seed2)
                best_arr = seed2

    # final safety: ensure validity
    if not _is_valid(best_arr):
        # fallback to a known-valid seed
        best_arr = _greedy_seed()

    return best_arr.tolist()
