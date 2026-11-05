import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # Safety margin
    
    # Miller-Rabin Primality Test
    # Bases sufficient for n < 3,317,444,400,000,000,000
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def phi(n):
        if n == 1:
            return 1
        factors = factorize(n)
        unique_factors = set(factors)
        result = n
        for p in unique_factors:
            result -= result // p
        return result

    best_k = 1
    best_s = 1

    def check_start(s):
        nonlocal best_k, best_s
        seen = set()
        k = 0
        # We can check up to 200 numbers
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            return True
        return False

    # Phase 1: Deterministic scan of small numbers
    # Small numbers are fast to factor but have higher collision probability.
    # Scan up to 100,000.
    scan_limit = 100000
    for s in range(1, scan_limit + 1):
        if time.time() > DEADLINE:
            break
        check_start(s)
        # Optimization: If we just updated best_k to 200, we can't do better.
        if best_k == 200:
            return {'start': best_s, 'length': best_k}
        
    # Phase 2: Random sampling in larger ranges
    # We sample starting points in [10^6, 10^12]
    # We need to be careful about the time spent on factorization.
    # Each check_start takes up to 200 phi calculations.
    # With Pollard Rho, phi(n) for n < 10^12 is very fast (microseconds to low milliseconds).
    # Let's estimate: 1000 random starts * 200 numbers * 1ms = 200s. Too slow if unlucky.
    # But average is much faster. We will monitor time.
    
    random.seed(42)
    
    # We will sample starts in chunks.
    # To avoid bias, we pick random points.
    # We also try to "walk" from a good point? No, random is safer for finding isolated long runs.
    
    # Let's try a structured random search.
    # We will pick random offsets in large intervals.
    
    while time.time() < DEADLINE - 10.0:
        # Pick a random start point
        # Avoid very small numbers (already scanned)
        s = random.randint(1000000, 10**12 - 200)
        
        # Optional: Check if this is near a previous check? 
        # With 10^12 space, collisions of start points are negligible.
        
        check_start(s)
        
        if best_k == 200:
            break
            
        # If we found a long run (e.g., > 50), we might want to explore neighbors?
        # But the task asks for the max k. A run of length k implies s..s+k-1 is valid.
        # s-1 might extend it. s+k might extend it.
        # Let's do a local optimization around the best found so far.
        if best_k > 50 and random.random() < 0.5:
            # Explore around best_s
            for delta in range(-10, 11):
                if delta == 0: continue
                new_s = best_s + delta
                if new_s >= 1 and new_s + 200 <= 10**12:
                    check_start(new_s)
                    if best_k == 200:
                        break
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}

# Keep the function name and signature as required
# The function is construct()
