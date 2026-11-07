import math
import time
from collections import defaultdict
import numpy as np

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin below 150s wall-clock

    # Fallback: a known small example with 3 solutions.
    # M = 333312, k in {336, 372, 434}
    def _fallback():
        M = 333312
        ks = [336, 372, 434]
        # verify quickly
        ok = []
        for k in ks:
            if k * _sigma_slow(k) == M:
                ok.append(k)
        if len(ok) >= 1:
            return {'M': int(M), 'k': [int(x) for x in ok]}
        # absolute minimal: k=1 -> M=1, sigma(1)=1, 1*1=1
        return {'M': 1, 'k': [1]}

    # Try a fast sigma sieve using NumPy vectorization.
    N = 10_000_000
    # sigma array (use uint64 for large values, but k*sigma(k) can exceed 2^64)
    # Actually, sigma(k) for k up to 10^7 is at most ~sigma(9979200) ~ 45M or so
    # k * sigma(k) can be up to 10^7 * 45*10^6 = 4.5 * 10^14, well within uint64
    sigma = np.zeros(N + 1, dtype=np.uint64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        if time.time() > deadline - 10.0:
            break

    if time.time() > deadline - 8.0:
        return _fallback()

    # Build map from M -> list of k using vectorized operations
    # Convert to Python list of M values
    ks_arr = np.arange(1, N + 1, dtype=np.uint64)
    M_vals = ks_arr * sigma[1:]  # k * sigma(k) for k = 1..N
    
    # Filter to M <= 10^18
    mask = M_vals <= 10**18
    M_valid = M_vals[mask]
    k_valid = ks_arr[mask]
    
    # Build map from M -> list of k
    M_to_ks = defaultdict(list)
    for i in range(len(M_valid)):
        M_to_ks[int(M_valid[i])].append(int(k_valid[i]))
        if time.time() > deadline - 8.0:
            break

    # Pick the M with the most k's (with k <= sqrt(M))
    best_M = None
    best_sols = None
    for M_val, ks in M_to_ks.items():
        if len(ks) < 2:
            continue
        limit = math.isqrt(M_val)
        valid = [k for k in ks if k <= limit]
        if best_sols is None or len(valid) > len(best_sols):
            best_sols = valid
            best_M = M_val
        if time.time() > deadline - 6.0:
            break

    if best_sols is None or len(best_sols) < 2:
        return _fallback()

    # Final exact verification with the sieve values (k <= N guaranteed)
    verified = []
    for k in best_sols:
        if 1 <= k <= math.isqrt(best_M) and k <= N:
            if int(k) * int(sigma[k]) == best_M:
                verified.append(k)

    if not verified:
        return _fallback()

    verified = sorted(set(verified))
    return {'M': int(best_M), 'k': [int(x) for x in verified]}
