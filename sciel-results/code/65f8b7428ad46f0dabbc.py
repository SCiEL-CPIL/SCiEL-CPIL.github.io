import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin below 150s limit

    # 1. Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 3,317,044,064,679,887,385,961,981
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # 2. Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g, r, q = 1, 1, 1
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

    # 3. Generate primes up to 10^7 using a simple sieve
    # This provides both candidate primes p and a list for trial division.
    limit = 10**7
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime_flag in enumerate(sieve) if is_prime_flag]

    # 4. Helper: Check if n is p-smooth
    # Optimized: Trial divide by small primes first, then use Pollard-Brent for the remainder.
    def is_p_smooth(n, p, small_primes):
        if n <= 1:
            return True
        if n <= p:
            return True # All factors of n are <= n <= p
        
        # Divide by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                if n <= p:
                    return True
                # If n is now prime, check if <= p
                if is_prime(n):
                    return n <= p
                # If n is composite and > p, we might need more factors
                # But if we've divided by small primes, n is likely a product of larger primes
                # or a prime > p.
                # We break out to handle the rest with Pollard-Brent or recursion if needed.
                # Actually, if n > p and we haven't reduced it to 1 or a prime <= p, 
                # it might still be p-smooth if it has factors <= p that are larger than our small_primes list.
                # So we cannot just return False here. We must factor the remainder.
                break
        
        if n == 1:
            return True
        if n <= p:
            return True
            
        # Check if remainder is prime
        if is_prime(n):
            return n <= p
        
        # Use Pollard-Brent to find a factor
        def check_factor(m, p):
            if m == 1:
                return True
            if m <= p:
                return True
            if is_prime(m):
                return m <= p
            
            # Try to find a factor
            f = None
            for _ in range(10): # Limit attempts
                f = pollard_brent(m)
                if f is not None and 1 < f < m:
                    break
            if f is None or f == m:
                return False
            
            return check_factor(f, p) and check_factor(m // f, p)
            
        return check_factor(n, p)

    # Precompute small primes for trial division (up to 10,000 is usually enough to strip most factors)
    small_primes = [q for q in primes if q <= 10000]

    # Main search loop
    best_k = 0
    best_p = 2
    
    # Strategy:
    # 1. Scan small primes to find any runs (likely short, but cheap).
    # 2. Randomly sample large primes to find potentially longer runs.
    #    The "smoothness sieve" idea is partially implemented by the trial division 
    #    in is_p_smooth which quickly rejects numbers with large prime factors.
    #    We also use the sieve to generate candidate primes efficiently.
    
    # Phase 1: Scan primes up to 10^5
    for p in primes:
        if time.time() > deadline:
            break
        if p > 100000:
            break
            
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            n = p * p + k + 1
            if is_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search for larger primes
    # We want to find larger k.
    # The expected number of primes to check for k >= K is ~ 1 / (0.3^K)
    # For K=10, ~ 1/5e-6 = 200,000 primes.
    # We can probably check ~ 10,000 - 50,000 large primes in the remaining time.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Pick a random prime in [10^5, 10^9]
        # To avoid bias, we can pick a random number and find the next prime.
        # Or we can pick a random prime from the sieve if we extend it, but 10^7 is enough for candidates.
        # Let's just pick random odd numbers in the range and test primality.
        
        lower = 100000
        upper = 10**9
        
        p_candidate = random.randint(lower, upper)
        if p_candidate % 2 == 0:
            p_candidate += 1
            
        # Find next prime
        # Miller Rabin is fast.
        while not is_prime(p_candidate):
            p_candidate += 2
            if p_candidate > upper:
                break
                
        if p_candidate > upper:
            continue
            
        # Check smoothness run
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            n = p_candidate * p_candidate + k + 1
            if is_p_smooth(n, p_candidate, small_primes):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p_candidate

    return {'p': best_p, 'k': best_k}
