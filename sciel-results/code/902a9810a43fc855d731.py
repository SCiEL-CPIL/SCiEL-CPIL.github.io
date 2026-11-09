import numpy as np
import time
import math
import random

N = 18
MASK_ALL = (1 << N) - 1
TIME_LIMIT = 85.0  # safety margin before the 90s kill

# ---------------------------------------------------------------------------
# Fast vectorized verifier (the proven ~100x speedup: one batch over all 2^N
# Boolean inputs instead of a Python loop).
# ---------------------------------------------------------------------------
def _is_sorted_network(comps, N):
    if N <= 1:
        return True
    if len(comps) == 0 and N > 1:
        return False
    inputs = np.arange(2 ** N, dtype=np.uint32)
    bits = 1 << np.arange(N, dtype=np.uint32)
    matrix = (inputs[:, None] & bits[None, :]) > 0
    for i, j in comps:
        mi = matrix[:, i]
        mj = matrix[:, j]
        mn = mi & mj
        mx = mi | mj
        matrix[:, i] = mn
        matrix[:, j] = mx
    diff = np.diff(matrix, axis=1)
    return bool(np.all(diff >= 0))

# ---------------------------------------------------------------------------
# Seed generators.
# ---------------------------------------------------------------------------
def _odd_even_transposition(n):
    nets = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                nets.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                nets.append((i, i + 1))
    return nets

def _bitonic_merge(n):
    if n <= 1:
        return []
    half = n // 2
    comps = [(i, i + half) for i in range(half)]
    left = _bitonic_merge(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    return comps + left + right

def _bitonic_sort(n):
    if n <= 1:
        return []
    half = n // 2
    left = _bitonic_sort(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    merge = _bitonic_merge(half)
    return left + right + merge

def _generate_bitonic_for_n(n):
    if n <= 1:
        return []
    p2 = 1
    while p2 < n:
        p2 *= 2
    full = _bitonic_sort(p2)
    return [(i, j) for (i, j) in full if i < n and j < n]

def _batcher_merge(N1, N2):
    if N1 == 0 or N2 == 0:
        return []
    depth = math.ceil(math.log2(N1 + N2))
    comps = []
    for j in range(depth):
        step = 1 << j
        for i in range(N1 + N2 - step):
            if j == 0:
                comps.append((i, i + step))
            else:
                if (i % step) < (step // 2):
                    comps.append((i, i + step))
    return comps

def _batcher_sort(N):
    if N == 1:
        return []
    N1 = N // 2
    N2 = N - N1
    comps = []
    comps.extend(_batcher_sort(N1))
    half2_comps = _batcher_sort(N2)
    comps.extend([(i + N1, j + N1) for i, j in half2_comps])
    comps.extend(_batcher_merge(N1, N2))
    return comps

def _dedup(comps):
    seen = set()
    out = []
    for c in comps:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# ---------------------------------------------------------------------------
# Local search: removal, insertion and swap moves, all verified with the fast
# batch verifier.  This is the structural change: a full neighborhood instead
# of the current removal-only pass.
# ---------------------------------------------------------------------------
def _local_search(net, deadline, rng):
    net = list(net)
    best = list(net)
    best_len = len(net)

    def verify(c):
        return _is_sorted_network(c, N)

    # ---- Phase A: removal (multi-pass, random order) ----
    while time.time() < deadline:
        order = list(range(len(net)))
        rng.shuffle(order)
        changed = False
        for idx in order:
            if time.time() > deadline:
                break
            if idx >= len(net):
                continue
            trial = net[:idx] + net[idx + 1:]
            if verify(trial):
                net = trial
                best = list(net)
                best_len = len(net)
                changed = True
        if not changed:
            break

    # ---- Phase B: swap (reorder) ----
    while time.time() < deadline:
        changed = False
        for _ in range(20000):
            if time.time() > deadline:
                break
            if len(net) < 2:
                break
            a = rng.randrange(len(net))
            b = rng.randrange(len(net))
            if a == b:
                continue
            net[a], net[b] = net[b], net[a]
            if verify(net):
                best = list(net)
                best_len = len(net)
                changed = True
            else:
                net[a], net[b] = net[b], net[a]
        if not changed:
            break

    # ---- Phase C: insertion (add a comparator, then prune) ----
    while time.time() < deadline:
        improved = False
        for _ in range(150):
            if time.time() > deadline:
                break
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            cand = list(net)
            pos = rng.randrange(len(cand) + 1)
            cand.insert(pos, (i, j))
            if verify(cand):
                net = cand
                # try to prune the freshly added network down to minimal
                order = list(range(len(net)))
                rng.shuffle(order)
                for idx in order:
                    if time.time() > deadline:
                        break
                    if idx >= len(net):
                        continue
                    trial = net[:idx] + net[idx + 1:]
                    if verify(trial):
                        net = trial
                if len(net) < best_len:
                    best = list(net)
                    best_len = len(net)
                improved = True
        if not improved:
            break

    return best, best_len

def construct():
    start_time = time.time()
    rng = random.Random(12345)

    # 1. Build candidate seeds and keep the smallest that verifies.
    candidates = []
    try:
        candidates.append(_generate_bitonic_for_n(N))
    except Exception:
        pass
    try:
        candidates.append(_batcher_sort(N))
    except Exception:
        pass
    candidates.append(_odd_even_transposition(N))

    best_net = None
    best_len = 10 ** 9
    for c in candidates:
        c = _dedup(c)
        if not c:
            continue
        if len(c) >= best_len:
            continue
        if _is_sorted_network(c, N):
            best_net = c
            best_len = len(c)
    if best_net is None:
        best_net = _odd_even_transposition(N)
        best_len = len(best_net)

    # 2. Local search from the best seed.
    deadline = start_time + TIME_LIMIT
    try:
        net, ln = _local_search(best_net, deadline, rng)
        if ln < best_len:
            best_net, best_len = net, ln
    except Exception:
        pass

    # 3. Safety: guarantee a valid fallback.
    if not _is_sorted_network(best_net, N):
        best_net = _odd_even_transposition(N)

    return best_net
