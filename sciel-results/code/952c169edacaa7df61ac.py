import math
import time

def construct():
    deadline = time.time() + 145  # 145s safety margin (evaluator kills at 150s)
    
    # Deterministic Miller-Rabin for n < 2^64
    _mr_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in _mr_bases:
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
    
    # Pollard's rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = 2
            c = 1
            m = 128
            g = 1
            q = 1
            r = 1
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
    
    def factorize(n, factors=None):
        """Return list of prime factors (with multiplicity)."""
        if factors is None:
            factors = []
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        d = pollard_brent(n)
        factorize(d, factors)
        factorize(n // d, factors)
        return factors
    
    def largest_prime_factor(n):
        """Return the largest prime factor of n."""
        if n == 1:
            return 1
        # Trial division by small primes first
        temp = n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while temp % p == 0:
                temp //= p
        if temp == 1:
            return 37  # largest small prime we tried
        if is_prime(temp):
            return max(temp, 37)
        factors = factorize(temp)
        return max(max(factors), 37)
    
    def run_length(p, max_k=200):
        """Find the length of the run starting at p^2+1 where all factors <= p."""
        k = 0
        p2 = p * p
        while k < max_k:
            n = p2 + k + 1
            if largest_prime_factor(n) > p:
                break
            k += 1
        return k
    
    # Pre-sieve: for a candidate p, check that p^2 + i (for i=1..limit) 
    # has no prime factor > p by checking divisibility by small primes.
    # If p^2 + i is divisible by a prime q > p, then it fails.
    # But we can't easily check that without factoring. Instead, we use:
    # For small primes q <= p, p^2 + i mod q == (p mod q)^2 + i mod q.
    # If for some i in [1, k_max], ALL small primes q that could divide p^2+i 
    # are <= p, we still need full factorization.
    # Better approach: just use fast trial division + Miller-Rabin for the check.
    
    # Strategy: iterate over primes p in increasing order, compute run length.
    # Use a fast primality test and efficient factorization.
    
    # Generate primes up to some limit using sieve
    SIEVE_LIMIT = 1000000  # primes up to 10^6
    
    def sieve(n):
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, n + 1) if is_p[i]]
    
    primes_list = sieve(SIEVE_LIMIT)
    
    best_k = 0
    best_p = 3
    
    # Phase 1: Check primes up to SIEVE_LIMIT
    for p in primes_list:
        if time.time() > deadline:
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check larger primes (between SIEVE_LIMIT and 10^9)
    # For larger p, we need to find primes. Use a segmented approach or
    # just test candidates.
    
    # For large p, the run length is likely small unless p is special.
    # But let's check some larger primes.
    
    # Generate more primes using a simple method for the range
    # We'll test odd numbers and use Miller-Rabin
    
    start = SIEVE_LIMIT + 1
    if start % 2 == 0:
        start += 1
    
    # Check primes in larger ranges, but be selective
    # Focus on ranges where we might find longer runs
    # Try primes in [10^6, 10^7], [10^7, 10^8], etc.
    
    for start_range in [SIEVE_LIMIT + 1, 10_000_000, 50_000_000, 100_000_000, 500_000_000]:
        if time.time() > deadline - 5:
            break
        if start_range > 10**9:
            break
        end_range = min(start_range + 5_000_000, 10**9)
        
        # Sieve this range
        low = start_range
        if low % 2 == 0:
            low += 1
        size = end_range - low + 1
        if size <= 0:
            continue
        
        # Use small primes to sieve
        small_primes = [p for p in primes_list if p * p <= end_range]
        
        # Simple check: test candidates
        n = low
        while n <= end_range and time.time() < deadline - 2:
            if is_prime(n):
                k = run_length(n)
                if k > best_k:
                    best_k = k
                    best_p = n
            n += 2
    
    return {'p': best_p, 'k': best_k}
