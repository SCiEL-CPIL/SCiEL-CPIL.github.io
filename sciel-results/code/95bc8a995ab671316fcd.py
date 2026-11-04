"""Sums/differences of finite sets: polynomial-support model + greedy coordinate search."""
import time
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0
_SEED = 20240611


def _score(u):
    """Lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not u:
        return 0.0
    u = sorted(set(int(x) for x in u))
    if 0 not in u:
        u = [0] + u
    n = len(u)
    if n < 2:
        return 0.0
    arr = np.array(u, dtype=np.int64)
    maxU = int(arr[-1])
    if maxU <= 0:
        return 0.0
    # |U+U| = |{a+b: a,b in U}|
    sums = arr[:, None] + arr[None, :]
    sumset = np.unique(sums).size
    # |U-U| = |{a-b: a,b in U}|
    diffs = arr[:, None] - arr[None, :]
    diffset = np.unique(diffs).size
    denom = np.log(2.0 * maxU + 1.0)
    if denom <= 0:
        return 0.0
    val = 1.0 + np.log(float(diffset) / float(sumset)) / denom
    return float(val)


def _init_candidates():
    """A diverse set of starting sets (all contain 0)."""
    cands = []
    # seed
    cands.append([0, 1, 3])
    # powers of 2
    cands.append([0] + [1 << i for i in range(1, 10)])
    # powers of 3
    cands.append([0] + [3 ** i for i in range(1, 8)])
    # Mersenne-ish
    cands.append([0] + [2 ** i - 1 for i in range(2, 12)])
    # random base-2 subsets
    rng = np.random.default_rng(_SEED)
    for _ in range(12):
        L = int(rng.integers(14, 26))
        bits = rng.random(L) < 0.45
        if not bits[0]:
            bits[0] = True
        vals = []
        v = 0
        for i, b in enumerate(bits):
            if b:
                v |= (1 << i)
                vals.append(v)
        vals = sorted(set(vals))
        if 0 not in vals:
            vals = [0] + vals
        cands.append(vals)
    # random sparse arithmetic-ish
    for _ in range(8):
        L = int(rng.integers(10, 20))
        step = int(rng.integers(2, 6))
        vals = [0] + list(range(step, step * L, step))
        # jitter some
        for i in range(1, len(vals)):
            if rng.random() < 0.3:
                vals[i] += int(rng.integers(-1, 2))
        vals = sorted(set(max(0, v) for v in vals))
        if 0 not in vals:
            vals = [0] + vals
        cands.append(vals)
    return cands


def _greedy(u, deadline):
    """Greedy coordinate search: try adding/removing each candidate element; keep best."""
    arr = np.array(sorted(set(u)), dtype=np.int64)
    cur = _score(arr.tolist())
    if not np.isfinite(cur):
        return arr.tolist(), cur
    best_u = arr.tolist()
    best_s = cur
    # candidate pool: multiples of small bases around existing range + small ints
    maxU = int(arr[-1]) if len(arr) else 0
    pool = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    for b in (2, 3, 4, 5):
        k = 1
        while b ** k <= 4 * maxU + 64:
            pool.add(b ** k)
            k += 1
    for v in arr:
        for d in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 120, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048):
            pool.add(v + d)
            if v >= d:
                pool.add(v - d)
    pool = sorted(set(max(0, int(x)) for x in pool))
    # cap pool
    if len(pool) > 800:
        pool = pool[:800]
    while time.time() < deadline:
        improved = False
        for x in pool:
            if time.time() >= deadline:
                break
            if x in set(best_u):
                # try removal (but keep 0 and size>=2)
                if x != 0 and len(best_u) > 2:
                    trial = [v for v in best_u if v != x]
                    s = _score(trial)
                    if s > best_s + 1e-12:
                        best_u = trial
                        best_s = s
                        improved = True
            else:
                trial = sorted(set(best_u) | {x})
                s = _score(trial)
                if s > best_s + 1e-12:
                    best_u = trial
                    best_s = s
                    improved = True
        if not improved:
            # expand pool a bit and try again
            maxU = int(best_u[-1]) if best_u else 0
            extra = set()
            for b in (2, 3, 5):
                k = 1
                while b ** k <= 8 * maxU + 256:
                    extra.add(b ** k)
                    k += 1
            for v in best_u:
                for d in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 37, 41, 47, 53, 61, 67, 71, 79, 83, 89, 97, 101, 127, 131, 151, 163, 181, 191, 199, 211, 223, 227, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997):
                    extra.add(v + d)
                    if v >= d:
                        extra.add(v - d)
            pool = sorted(set(pool) | set(max(0, int(x)) for x in extra))
            if len(pool) > 1600:
                pool = pool[:1600]
            if not improved and len(pool) >= 1600:
                break
    return best_u, best_s


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    cands = _init_candidates()
    best_u = None
    best_s = -1.0
    for c in cands:
        if time.time() >= deadline:
            break
        s = _score(c)
        if s > best_s:
            best_s = s
            best_u = c
    # greedy refine top few
    ranked = []
    for c in cands:
        if time.time() >= deadline:
            break
        s = _score(c)
        ranked.append((s, c))
    ranked.sort(key=lambda t: -t[0])
    for s, c in ranked[:4]:
        if time.time() >= deadline:
            break
        u2, s2 = _greedy(c, deadline)
        if s2 > best_s:
            best_s = s2
            best_u = u2
    if best_u is None:
        best_u = [0, 1, 3]
    # final safety: ensure valid
    try:
        if not (0 in best_u and len(best_u) >= 1):
            best_u = [0, 1, 3]
        return [int(x) for x in best_u]
    except Exception:
        return [0, 1, 3]
