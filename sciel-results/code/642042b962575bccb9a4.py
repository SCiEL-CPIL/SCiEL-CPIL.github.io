import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0

    def elapsed():
        return time.time() - start_time

    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Factorize n using trial division + Miller-Rabin
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n and d <= 10**6:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            if is_prime(n):
                factors[n] = factors.get(n, 0) + 1
            else:
                # n is composite but large - try to find a factor
                # For our purposes, if n is a perfect square of a prime
                r = int(math.isqrt(n))
                if r * r == n and is_prime(r):
                    factors[r] = factors.get(r, 0) + 2
                else:
                    factors[n] = factors.get(n, 0) + 1
        return factors

    # Compute sigma from factorization dict {p: exp}
    def sigma_from_factors(factors):
        result = 1
        for p, e in factors.items():
            s = 1
            pe = 1
            for _ in range(e):
                pe *= p
                s += pe
            result *= s
        return result

    # Sieve up to N
    N = 3000000
    spf = list(range(N + 1))
    if N >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute sigma for all k up to N
    sigma = [0] * (N + 1)
    sigma[1] = 1
    for k in range(2, N + 1):
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
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            M_to_ks[M].append(k)

    # Find candidates with count >= 2
    candidates = [(M, ks) for M, ks in M_to_ks.items() if len(ks) >= 2]
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    best_M = None
    best_ks = []

    # For top candidates, try to extend
    top_candidates = candidates[:100]

    for M, ks in top_candidates:
        if elapsed() > deadline - 20:
            break

        max_k = math.isqrt(M)
        if max_k <= N:
            continue

        # Factorize M
        temp_M = M
        factors = {}
        d = 2
        while d * d <= temp_M and d <= 10**6:
            while temp_M % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp_M //= d
            d += 1 if d == 2 else 2
        if temp_M > 1:
            if is_prime(temp_M):
                factors[temp_M] = factors.get(temp_M, 0) + 1
            else:
                r = int(math.isqrt(temp_M))
                if r * r == temp_M and is_prime(r):
                    factors[r] = factors.get(r, 0) + 2
                else:
                    factors[temp_M] = factors.get(temp_M, 0) + 1

        # Generate all divisors of M
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divisors:
                pe = 1
                for e in range(exp + 1):
                    new_divs.append(d * pe)
                    pe *= p
            divisors = new_divs

        # Filter divisors k with N < k <= max_k
        existing = set(ks)
        for k in divisors:
            if N < k <= max_k and k not in existing:
                # Compute sigma(k)
                if k <= N:
                    sig = sigma[k]
                else:
                    kf = factorize(k)
                    sig = sigma_from_factors(kf)
                if sig == M // k:
                    ks.append(k)
                    existing.add(k)

        if len(ks) > len(best_ks):
            best_ks = ks[:]
            best_M = M

    if best_M is None:
        for M, ks in candidates:
            if len(ks) > len(best_ks):
                best_ks = ks[:]
                best_M = M

    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
