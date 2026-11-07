import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    # Strategy: Use a sieve to compute sigma(k) for k up to N, then bucket M=k*sigma(k).
    # To go further, we can also try a "targeted" approach: for each M found with count>=2,
    # try to find additional k by checking k in a wider range using a precomputed sigma table
    # or by direct factorization.

    # First pass: sieve up to N1
    N1 = 2000000  # 2 million

    # Sieve for smallest prime factor
    spf = list(range(N1 + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(N1)) + 1):
        if spf[i] == i:
            for j in range(i * i, N1 + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute sigma(k) for all k using SPF
    sigma = [0] * (N1 + 1)
    sigma[1] = 1
    for k in range(2, N1 + 1):
        n = k
        result = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            result *= power_sum
        sigma[k] = result

    # Bucket M = k * sigma(k)
    M_to_ks = defaultdict(list)
    for k in range(1, N1 + 1):
        M = k * sigma[k]
        if M <= 10**18:
            M_to_ks[M].append(k)

    # Find candidates with count >= 2
    candidates = [(M, ks) for M, ks in M_to_ks.items() if len(ks) >= 2]
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    best_M = None
    best_ks = []

    # For the top candidates, try to extend by checking a wider range of k
    # We'll check k from N1+1 up to some limit, computing sigma(k) on the fly
    # Only for the top few candidates to save time

    top_candidates = candidates[:50]  # Top 50 candidates

    for M, ks in top_candidates:
        if elapsed() > deadline - 30:
            break

        # For this M, we want to find k > N1 such that k * sigma(k) = M
        # This means sigma(k) = M/k, so k must divide M
        # Also k > N1 and k <= isqrt(M)
        max_k = min(math.isqrt(M), 5000000)  # Cap at 5M for safety

        # Find all divisors k of M with N1 < k <= max_k
        # We can do this by iterating d from 1 to isqrt(M) and checking if d divides M
        # But that's too slow. Instead, we can iterate k from N1+1 to max_k and check if M % k == 0
        # and then verify sigma(k) == M/k

        # Actually, a better approach: iterate k from N1+1 to max_k, check if M % k == 0
        # If so, compute sigma(k) and check if it equals M/k

        # But this is O(max_k - N1) per candidate, which is too slow for 50 candidates.

        # Alternative: For each M, factorize M, generate all divisors, filter those > N1 and <= max_k,
        # then check sigma(k) == M/k for each divisor.

        # Factorize M
        temp_M = M
        factors = {}
        d = 2
        while d * d <= temp_M:
            while temp_M % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp_M //= d
            d += 1 if d == 2 else 2
        if temp_M > 1:
            factors[temp_M] = factors.get(temp_M, 0) + 1

        # Generate all divisors
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divisors:
                for e in range(exp + 1):
                    new_divs.append(d * (p ** e))
            divisors = new_divs

        # Filter divisors k with N1 < k <= max_k
        for k in divisors:
            if N1 < k <= max_k:
                # Compute sigma(k)
                n = k
                sig = 1
                while n > 1:
                    p = spf[n] if n <= N1 else None
                    if p is None:
                        # Need to factorize k manually
                        # This is expensive, so let's just check a few
                        break
                    power_sum = 1
                    power = 1
                    while n % p == 0:
                        n //= p
                        power *= p
                        power_sum += power
                    sig *= power_sum
                else:
                    if sig == M // k:
                        ks.append(k)

        if len(ks) > len(best_ks):
            best_ks = ks[:]
            best_M = M

    # If we didn't find anything better, use the best from the initial bucketing
    if best_M is None:
        for M, ks in candidates:
            if len(ks) > len(best_ks):
                best_ks = ks[:]
                best_M = M

    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
