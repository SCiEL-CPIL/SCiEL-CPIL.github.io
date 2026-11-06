import random
import math

def construct():
    # 1. Sieve primes up to 10^6
    limit = 10**6
    is_prime_sieve = [True] * (limit + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_sieve[j] = False
    primes = [i for i, val in enumerate(is_prime_sieve) if val]
    
    # 2. Miller-Rabin
    def is_prime_mr(n):
        if n < 2: return False
        if n in (2, 3): return True
        if n % 2 == 0: return False
        # Deterministic witnesses for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in witnesses:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Fast Phi
    def phi(n):
        if n == 1: return 1
        if is_prime_mr(n): return n - 1
        
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result

    # 4. Search
    best_k = 1
    best_s = 1
    
    # Search small range systematically
    # We can go up to 2 * 10^6 easily
    N_SMALL = 2 * 10**6
    values = [0] * (N_SMALL + 1)
    for i in range(1, N_SMALL + 1):
        values[i] = phi(i)
        
    for s in range(1, N_SMALL):
        seen = set()
        k = 0
        while s + k <= N_SMALL and values[s + k] not in seen:
            seen.add(values[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            
    # Random search in large range
    # If best_k is already high, we might not need many samples
    # But let's try to find even longer runs
    for _ in range(100):
        s = random.randint(1, 10**12)
        seen = set()
        k = 0
        max_k_to_check = min(200, 10**12 - s + 1)
        for i in range(max_k_to_check):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            
    return {'start': best_s, 'length': best_k}
