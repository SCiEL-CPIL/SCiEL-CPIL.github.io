import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        # Witnesses for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in witnesses:
            if a >= n:
                continue
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

    # Pollard Rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = 128
            g, q, r = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    # Precompute small primes for trial division
    # Limit 10000 is a good balance. Primes up to 10000: 1229 primes.
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        
        # 1. Strip small primes
        temp_n = n
        for q in small_primes:
            if q > p:
                break
            if temp_n % q == 0:
                while temp_n % q == 0:
                    temp_n //= q
                if temp_n == 1:
                    return True
                if temp_n <= p:
                    return True
        
        if temp_n == 1:
            return True
            
        if temp_n <= p:
            return True
            
        # 2. If temp_n > p, check if prime
        if is_prime_mr(temp_n):
            return False  # temp_n is a prime factor > p
        
        # 3. temp_n is composite and > p
        # Find a factor using Pollard-Brent
        d = pollard_brent(temp_n)
        if d == temp_n:
            # Pollard failed, try again with different params
            d = pollard_brent(temp_n)
        return is_p_smooth(d, p) and is_p_smooth(temp_n // d, p)

    best_k = 0
    best_p = 2

    # Phase 1: Sieve up to 10^6
    max_p_search = 1000000
    sieve_large = bytearray(b'\x01') * (max_p_search + 1)
    sieve_large[0] = sieve_large[1] = 0
    for i in range(2, int(max_p_search**0.5) + 1):
        if sieve_large[i]:
            for j in range(i*i, max_p_search + 1, i):
                sieve_large[j] = 0
    
    primes_to_check = [i for i in range(2, max_p_search + 1) if sieve_large[i]]
    
    # Quick pre-filter: check if p^2+1 is p-smooth before doing full check
    def quick_check_p_smooth(n, p):
        """Quick check: strip small primes, see if remainder is <= p or prime"""
        temp_n = n
        for q in small_primes:
            if q > p:
                break
            if temp_n % q == 0:
                while temp_n % q == 0:
                    temp_n //= q
        if temp_n == 1 or temp_n <= p:
            return True
        # If temp_n is prime > p, definitely not smooth
        if is_prime_mr(temp_n):
            return False
        return True  # composite, need full check
    
    for p in primes_to_check:
        if time.time() > deadline:
            break
        base = p * p
        # Quick reject if p^2+1 is not p-smooth
        if not quick_check_p_smooth(base + 1, p):
            continue
        k = 0
        for i in range(1, 201):
            n = base + i
            if not is_p_smooth(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': p, 'k': 200}

    # Phase 2: Scan larger primes
    # We'll scan up to 10^7 or more if time permits.
    # Use Miller-Rabin to find primes.
    current = max_p_search + 1
    if current % 2 == 0:
        current += 1
        
    # Scan up to 10,000,000
    SCAN_LIMIT = 10_000_000
    while current <= SCAN_LIMIT:
        if time.time() > deadline:
            break
        if is_prime_mr(current):
            p = current
            base = p * p
            # Quick reject
            if not quick_check_p_smooth(base + 1, p):
                current += 2
                continue
            k = 0
            for i in range(1, 201):
                n = base + i
                if not is_p_smooth(n, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    return {'p': p, 'k': 200}
        current += 2
        
    # Phase 3: Random sampling up to 10^9
    # If time remains, try random primes in larger ranges.
    # Focus on ranges where p^2+i values are more likely to be p-smooth
    while time.time() < deadline - 5:
        # Pick a random odd number in [10^7, 10^9]
        p_candidate = random.randint(10_000_000, 1_000_000_000)
        if p_candidate % 2 == 0:
            p_candidate += 1
            
        # Quick check for small factors
        is_prem = True
        for q in small_primes:
            if p_candidate % q == 0:
                if p_candidate == q:
                    break
                is_prem = False
                break
        if not is_prem:
            continue
            
        if is_prime_mr(p_candidate):
            p = p_candidate
            base = p * p
            # Quick reject
            if not quick_check_p_smooth(base + 1, p):
                continue
            k = 0
            for i in range(1, 201):
                n = base + i
                if not is_p_smooth(n, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    return {'p': p, 'k': 200}
                
    return {'p': best_p, 'k': best_k}
