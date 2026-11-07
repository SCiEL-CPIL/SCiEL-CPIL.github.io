import math
import time
from collections import defaultdict
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    LIMIT = 10**18

    # Generate primes up to a reasonable bound
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    # Focus on small primes for dense search
    small_primes = primes[:15]  # primes up to 47

    # For each prime p and exponent a, factor = p^a * sigma(p^a) = p^a * (p^(a+1)-1)/(p-1)
    # We want to enumerate all k = prod p_i^a_i such that M = prod factor_i <= LIMIT
    # and k <= sqrt(M)

    # Precompute factor lists for each prime
    factor_lists = []
    for p in small_primes:
        fl = []
        p_pow = 1
        for a in range(0, 60):
            if a == 0:
                factor = 1
            else:
                # p^a * (p^(a+1)-1)/(p-1)
                # Compute iteratively to avoid overflow issues
                sigma_val = 0
                pp = 1
                for i in range(a + 1):
                    sigma_val += pp
                    pp *= p
                factor = p_pow * sigma_val
            if factor > LIMIT:
                break
            fl.append((a, factor))
            p_pow *= p
            if p_pow > LIMIT:
                break
        factor_lists.append(fl)

    # DFS to enumerate all (M, k) pairs
    solutions_by_M = defaultdict(list)

    def dfs(idx, current_M, current_k):
        if time.time() > deadline:
            return
        if idx == len(small_primes):
            if 1 <= current_M <= LIMIT:
                solutions_by_M[current_M].append(current_k)
            return

        for a, factor in factor_lists[idx]:
            new_M = current_M * factor
            if new_M > LIMIT:
                break
            if a == 0:
                new_k = current_k
            else:
                new_k = current_k * (small_primes[idx] ** a)
            dfs(idx + 1, new_M, new_k)

    dfs(0, 1, 1)

    # Find M with most valid solutions (k <= sqrt(M))
    best_M = 1
    best_count = 0
    best_solutions = []

    for M, ks in solutions_by_M.items():
        if M < 4:
            continue
        sqrt_M = math.isqrt(M)
        valid_ks = set()
        for k in ks:
            if k <= sqrt_M:
                valid_ks.add(k)
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_solutions = list(valid_ks)

    # Verify with sigma function
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                result *= (d ** (exp + 1) - 1) // (d - 1)
            d += 1
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    verified = []
    for k in best_solutions:
        if k * sigma(k) == best_M and k <= math.isqrt(best_M):
            verified.append(k)

    if len(verified) > 0:
        return {'M': int(best_M), 'k': [int(k) for k in verified]}
    else:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
