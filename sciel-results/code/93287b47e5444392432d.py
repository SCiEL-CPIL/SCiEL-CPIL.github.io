"""Rewritten program: structured search for integer sum-product extremal set (n=20, max=10000)."""
import time
import random
import math

_N = 20
_MAX = 10000
_DEADLINE = 90.0
_SAFETY = 8.0


def _sumset_size(a):
    s = set()
    n = len(a)
    for i in range(n):
        ai = a[i]
        for j in range(i, n):
            s.add(ai + a[j])
    return len(s)


def _prodset_size(a):
    s = set()
    n = len(a)
    for i in range(n):
        ai = a[i]
        for j in range(i, n):
            s.add(ai * a[j])
    return len(s)


def _objective(a):
    return max(_sumset_size(a), _prodset_size(a))


def _smooth_candidates():
    cands = set()
    # 5-smooth (Hamming) numbers up to 10000
    i = j = k = 1
    while True:
        vals = [v for v in (i, j, k) if v <= _MAX]
        if not vals:
            break
        m = min(vals)
        cands.add(m)
        if m >= _MAX:
            break
        if m == i:
            i *= 2
        if m == j:
            j *= 2
        if m == k:
            k *= 2
    base = [2, 3, 5, 6, 10, 12, 15, 20, 24, 30, 40, 48, 60, 80, 120, 160, 240, 320,
            480, 640, 960, 1280, 1920, 2560, 3840, 5120, 7680]
    for b in base:
        m = b
        while m <= _MAX:
            cands.add(m)
            m *= 2
        m = b
        while m <= _MAX:
            cands.add(m)
            m *= 3
    # spread values for diversity
    for v in range(2, _MAX + 1):
        cands.add(v)
        if len(cands) > 4000:
            break
    return sorted(cands)


def _greedy_init(pool, rng):
    a = [1]
    used = {1}
    remaining = [x for x in pool if x not in used]
    for _ in range(_N - 1):
        best_val, best_obj = None, None
        sample = remaining if len(remaining) <= 400 else rng.sample(remaining, 400)
        for x in sample:
            obj = _objective(a + [x])
            if best_obj is None or obj < best_obj:
                best_obj, best_val = obj, x
        if best_val is None:
            best_val = remaining[0]
        a.append(best_val)
        used.add(best_val)
        remaining = [x for x in remaining if x != best_val]
    return a


def _local_search(a, pool, rng, deadline):
    cur = a[:]
    cur_obj = _objective(cur)
    best, best_obj = cur[:], cur_obj
    T = 6.0
    cool = 0.999
    used = set(cur)
    swap_pool = list(set(cur) | set(pool[:600]))
    n = len(cur)
    step = 0
    while time.time() < deadline:
        step += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        new_val = rng.choice(swap_pool)
        if new_val in used:
            continue
        old_val = cur[i]
        cur[i] = new_val
        used.discard(old_val)
        used.add(new_val)
        new_obj = _objective(cur)
        if new_obj <= cur_obj:
            cur_obj = new_obj
            if new_obj < best_obj:
                best_obj, best = new_obj, cur[:]
        else:
            if rng.random() < math.exp(-(new_obj - cur_obj) / max(T, 1e-9)):
                cur_obj = new_obj
            else:
                cur[i] = old_val
                used.discard(new_val)
                used.add(old_val)
        T *= cool
        if step % 500 == 0:
            extra = rng.sample(pool, min(200, len(pool)))
            swap_pool = list(set(cur) | set(extra))
    return best, best_obj


def _finalize(a):
    # guarantee: 20 distinct positive ints <= _MAX
    seen = set()
    out = []
    for x in a:
        if isinstance(x, int) and 1 <= x <= _MAX and x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) == _N:
            break
    if len(out) < _N:
        v = 1
        while len(out) < _N and v <= _MAX:
            if v not in seen:
                seen.add(v)
                out.append(v)
            v += 1
    return out[:_N]


def construct():
    rng = random.Random(12345)
    start = time.time()
    deadline = start + _DEADLINE - _SAFETY

    pool = _smooth_candidates()
    if len(pool) < 2 * _N:
        pool = list(range(1, 2 * _N + 1))

    a = _greedy_init(pool, rng)
    best, best_obj = a, _objective(a)

    if time.time() < deadline:
        cand, cand_obj = _local_search(a, pool, rng, deadline)
        if cand_obj < best_obj:
            best, best_obj = cand, cand_obj

    return _finalize(best)
