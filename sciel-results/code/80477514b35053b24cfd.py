import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    # 1. Sieve of Eratosthenes for primes up to 10^6
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # 2. Precompute phi up to LIMIT
    phi = list(range(LIMIT + 1))
    for p in primes:
        for j in range(p, LIMIT + 1, p):
            phi[j] -= phi[j] // p
            
    # 3. Fast primality test and factorization for n > LIMIT
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Test bases sufficient for n < 3.3e24 (covers 10^12 easily)
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        # Use a deterministic-ish sequence of c values to avoid infinite loops
        # but randomize start points.
        for c in range(1, 10):
            x = 2
            y = 2
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
        # Fallback to trial division if rho fails (unlikely for 10^12 range)
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                return p
        return n

    def factor(n, factors=None):
        if factors is None:
            factors = {}
        if n == 1:
            return factors
        if n <= LIMIT:
            # Use precomputed phi? No, we need factors for phi formula if we were doing it manually,
            # but actually we have phi[n] directly.
            # However, this function is only called for n > LIMIT in get_phi.
            # Wait, recursion might call factor on smaller numbers?
            # If n <= LIMIT, we can just return phi[n] directly in get_phi.
            # So factor is only called for n > LIMIT.
            # But pollard_rho might return a factor that is <= LIMIT?
            # If d <= LIMIT, factor(d) should be efficient.
            # We can optimize by checking if n is prime first.
            pass
            
        if is_prime_mr(n):
            factors[n] = factors.get(n, 0) + 1
            return factors
            
        d = pollard_rho(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    def phi_large(n):
        facs = factor(n)
        result = n
        for p in facs:
            result -= result // p
        return result

    def get_phi(n):
        if n <= LIMIT:
            return phi[n]
        else:
            return phi_large(n)

    best_k = 1
    best_s = 1
    
    # 4. Scan small range (0 to LIMIT) using precomputed phi
    # This is extremely fast.
    # We scan up to LIMIT - 200 to ensure window fits.
    scan_limit = LIMIT - 200
    for s in range(1, scan_limit):
        if time.time() - start_time > DEADLINE:
            break
        seen = set()
        k = 0
        # Max k is 200
        while k < 200:
            v = phi[s + k]
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s

    # 5. Random search for larger s
    # If best_k is small, maybe larger s has larger k?
    # Theoretical max is around 27.6^c, but practically small.
    # We use random sampling to find rare long runs.
    
    max_n = 10**12
    random.seed(42)
    
    def check_window(s):
        seen = set()
        k = 0
        while k < 200 and s + k <= max_n:
            v = get_phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Strategy:
    # 1. Random samples across the range.
    # 2. If a sample yields k > best_k, do a local neighborhood search around it.
    # 3. Also, try some structured starts (e.g., multiples of primorials) where collisions are less likely?
    #    Actually, distinct totients are more likely when numbers have distinct prime factorizations.
    #    Random numbers are good.
    
    attempts = 0
    while time.time() - start_time < DEADLINE:
        attempts += 1
        s = random.randint(LIMIT, max_n - 200)
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
            # Local search around the new best
            # Check neighbors to see if we can extend or find a better one nearby
            for offset in range(-200, 201, 5):
                if time.time() - start_time > DEADLINE:
                    break
                ns = s + offset
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns

    return {'start': best_s, 'length': best_k}
