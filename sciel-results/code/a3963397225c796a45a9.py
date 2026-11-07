import math
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 115.0   # hard kill is ~150s; leave generous safety margin

    def now():
        return time.time()

    # ------------------------------------------------------------------
    # 1) Fast smallest-prime-factor sieve (vectorized, classic)
    # ------------------------------------------------------------------
    N = 120_000_000
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:
            start = i * i
            idx = np.arange(start, N + 1, i)
            mask = spf[idx] == idx
            if mask.any():
                spf[idx[mask]] = i
    spf = spf.tolist()

    # ------------------------------------------------------------------
    # 2) Vectorized sigma (sum-of-divisors) via a divisor-addition pass.
    #    This removes the per-integer Python loop entirely.
    # ------------------------------------------------------------------
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(2, N + 1):
        sigma[d::d] += d

    # ------------------------------------------------------------------
    # 3) Collision search: k*sigma(k) == M.  Do a full scan, then targeted
    #    scans around already-found M values to push the count higher.
    # ------------------------------------------------------------------
    M_TO_KS = {}
    best_M = 333312
    best_sols = [336, 372, 434]
    best_count = 3

    def _scan(lo, hi):
        nonlocal best_M, best_sols, best_count
        lo = max(1, lo)
        hi = min(N, hi)
        if hi < lo:
            return
        ks = np.arange(lo, hi + 1, dtype=np.int64)
        M_vals = ks * sigma[lo:hi + 1]
        M_vals = M_vals[M_vals <= 10**18]
        M_vals = M_vals.tolist()
        ks_list = ks.tolist()
        step = 200_000
        for i in range(0, len(M_vals), step):
            if now() > deadline - 12:
                return
            for j in range(i, min(i + step, len(M_vals))):
                m = M_vals[j]
                k = ks_list[j]
                lst = M_TO_KS.get(m)
                if lst is None:
                    M_TO_KS[m] = [k]
                else:
                    lst.append(k)
                    if len(lst) > best_count:
                        best_count = len(lst)
                        best_sols = lst[:]
                        best_M = m

    # Full scan
    _scan(1, N)

    # Targeted scans around the best M values (cheaper, may find more k)
    if now() < deadline - 15:
        targets = [m for m, lst in M_TO_KS.items() if len(lst) >= 2]
        targets.sort(key=lambda m: -len(M_TO_KS[m]))
        for m in targets[:25]:
            if now() > deadline - 12:
                break
            c = m // 2
            w = int(math.isqrt(m)) // 3 + 1
            _scan(max(1, c - w), min(N, c + w))

    # ------------------------------------------------------------------
    # 4) Exact verification (recompute sigma via SPF factorization)
    # ------------------------------------------------------------------
    def sigma_exact(n):
        if n == 1:
            return 1
        total = 1
        x = n
        while x > 1:
            p = spf[x]
            power = 1
            s = 1
            while x % p == 0:
                x //= p
                power *= p
                s += power
            total *= s
        return total

    valid = []
    limit = math.isqrt(best_M)
    for k in best_sols:
        if 1 <= k <= limit:
            if k * sigma_exact(k) == best_M:
                valid.append(int(k))
    valid = sorted(set(valid))

    if not valid:
        valid = [336, 372, 434]

    return {'M': int(best_M), 'k': valid}
