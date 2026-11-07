import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # hard safety margin before the 150s kill

    def _sigma_from_factors(factors):
        if not factors:
            return 1
        result = 1
        for p, e in factors.items():
            pe = pow(p, e + 1)
            result *= (pe - 1) // (p - 1)
        return result

    def _factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def _verify(M, ks):
        # Exact factorization check; keep only distinct k with k <= isqrt(M) and k*sigma(k)==M.
        isq = math.isqrt(M)
        out = []
        seen = set()
        for k in ks:
            kk = int(k)
            if kk in seen:
                continue
            if kk < 1 or kk > isq:
                continue
            if kk * _sigma_from_factors(_factorize(kk)) == M:
                out.append(kk)
                seen.add(kk)
        return sorted(out)

    def _best_valid(ks_by_M, M_arr, k_arr, limit=5000):
        # Scan candidate M values (in descending order of raw preimage count) and
        # return the first whose valid (k <= isqrt(M)) solutions are non-empty.
        unique_M, inverse = np.unique(M_arr, return_inverse=True)
        counts = np.bincount(inverse)
        order = np.argsort(counts)[::-1]
        checked = 0
        for idx in order:
            if checked >= limit:
                break
            if time.time() > deadline - 20.0:
                break
            checked += 1
            M = int(unique_M[idx])
            if M < 1 or M > 10**18:
                continue
            isq = math.isqrt(M)
            ks = k_arr[inverse == idx]
            ks = ks[ks <= isq]
            if len(ks) == 0:
                continue
            ks = ks.tolist()
            verified = _verify(M, ks)
            if verified:
                return M, verified
        return None, None

    def _fallback(K):
        # Smaller, self-contained sieve used if the main run is interrupted.
        if K < 200000:
            K = 200000
        sigma = np.zeros(K + 1, dtype=np.int64)
        for d in range(1, K + 1):
            if time.time() > deadline - 15.0:
                break
            sigma[d::d] += d
        k_arr = np.arange(1, K + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]
        valid_mask = (M_arr >= 1) & (M_arr <= 10**18)
        M, ks = _best_valid(M_arr[valid_mask], k_arr[valid_mask])
        if M is None:
            return None
        return {'M': int(M), 'k': [int(x) for x in ks][:1000]}

    # ---- Main sieve over a large range, using a cheap periodic time check ----
    K_max = 100_000_000
    sigma = np.zeros(K_max + 1, dtype=np.int64)
    interrupted = False
    t0 = time.time()
    for d in range(1, K_max + 1):
        if (d & 1023) == 0:
            if time.time() > deadline - 25.0:
                interrupted = True
                break
        sigma[d::d] += d
    if time.time() > deadline - 25.0:
        interrupted = True

    # Determine how much of the sieve is complete (the loop is monotonic in d).
    d_last = d if not interrupted else (d - 1)
    K = d_last
    if K < 200000:
        res = _fallback(200000)
        if res is not None:
            return res
        return {'M': 333312, 'k': [336, 372, 434]}

    k_arr = np.arange(1, K + 1, dtype=np.int64)
    M_arr = k_arr * sigma[1:K + 1]
    valid_mask = (M_arr >= 1) & (M_arr <= 10**18)
    M_arr = M_arr[valid_mask]
    k_arr = k_arr[valid_mask]

    if len(M_arr) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    M, ks = _best_valid(M_arr, k_arr)

    if M is not None and ks:
        ks = ks[:1000]
        return {'M': int(M), 'k': [int(x) for x in ks]}

    # If the large run found nothing valid, fall back to a smaller verified sieve.
    res = _fallback(5_000_000)
    if res is not None:
        return res
    return {'M': 333312, 'k': [336, 372, 434]}
