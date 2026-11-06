"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    # (covers all n we will test, since p <= 1e9, p^2+k <= ~1e18+200)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Small primes check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 = d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 3.3e24
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def factorize(n):
        """Return list of prime factors of n (with multiplicity)."""
        if n <= 1:
            return []
        factors = []
        # Trial division by small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        # Use Pollard's Brent
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                if d == m:
                    # Retry
                    stack.append(m)
                    continue
                stack.append(d)
                stack.append(m // d)
        return factors

    def largest_prime_factor(n):
        """Return the largest prime factor of n."""
        if n <= 1:
            return 1
        # Quick check: if n is prime, return n
        if is_prime(n):
            return n
        factors = factorize(n)
        return max(factors)

    def all_factors_leq(n, limit):
        """Check if all prime factors of n are <= limit. Returns True/False."""
        if n <= 1:
            return True
        # Trial division by small primes first
        temp = n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while temp % p == 0:
                temp //= p
        if temp == 1:
            return True
        if temp > limit:
            # temp has a factor > limit (since temp is not divisible by small primes)
            # But wait: temp might be composite with all factors <= limit but > 97
            # We need to fully factor temp
            pass
        # Check if temp is prime
        if is_prime(temp):
            return temp <= limit
        # Factor temp
        factors = factorize(temp)
        return all(f <= limit for f in factors)

    def smooth_run_length(p, max_k=200):
        """Find the longest run p^2+1, ..., p^2+k where all prime factors <= p."""
        k = 0
        base = p * p
        while k < max_k:
            if time.time() > deadline:
                return k
            val = base + k + 1
            if all_factors_leq(val, p):
                k += 1
            else:
                break
        return k

    # Strategy: Search over primes p, using a smoothness-aware approach
    # Key insight: For p^2 + j to be p-smooth, it must not have any prime factor > p.
    # We can pre-filter: if p^2 + j is divisible by a prime q > p, it fails.
    # 
    # Better approach: For each candidate p, compute the run length.
    # To find good p values, note that p^2 + 1, p^2 + 2, ... must all be p-smooth.
    # 
    # Let's try a range of primes. Start with small primes where runs might be longer,
    # but also try larger primes.
    
    best_p = 2
    best_k = 0

    # Generate primes up to a reasonable bound for search
    # We'll search p in a range. Let's be smart about which p to try.
    
    # First, try small primes (up to ~10000) since they're cheap to check
    # and may have longer relative runs
    
    def sieve_primes(limit):
        """Sieve of Eratosthenes."""
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Try primes in increasing order, but also jump to larger primes
    # The seed found best k=2 with small primes. Let's see if larger primes do better.
    
    # Phase 1: Try primes up to 50000
    primes_small = sieve_primes(50000)
    for p in primes_small:
        if time.time() > deadline:
            break
        k = smooth_run_length(p, 200)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Try larger primes in strategic ranges
    # Larger p means p^2 is larger, so p^2+j for small j might be harder to be p-smooth
    # but the "density" of primes > p in the range [p^2+1, p^2+200] depends on p.
    # For very large p, p^2+j is likely prime (or has a large factor) for most j.
    # Actually, for large p, p^2+j is roughly of size p^2, and the probability
    # that a random number of size p^2 has all prime factors <= p is related to
    # the smoothness probability.
    
    # Let's try some larger primes
    # Try primes around 10^5 to 10^6
    if time.time() < deadline - 10:
        # Generate primes in a few ranges
        for lo, hi in [(50000, 100000), (100000, 200000), (200000, 400000),
                        (400000, 700000), (700000, 1000000)]:
            if time.time() > deadline:
                break
            # Sample some primes in this range
            # Use a simple sieve for the range
            size = hi - lo
            if size < 2:
                continue
            # Simple prime test for numbers in [lo, hi]
            # This is expensive, so let's just sample
            import random as _random
            _random.seed(42)
            candidates = _random.sample(range(lo, hi), min(200, hi - lo))
            for p in candidates:
                if time.time() > deadline:
                    break
                if is_prime(p):
                    k = smooth_run_length(p, 200)
                    if k > best_k:
                        best_k = k
                        best_p = p

    # Phase 3: Try even larger primes if time permits
    if time.time() < deadline - 5:
        import random as _random
        _random.seed(123)
        for _ in range(50):
            if time.time() > deadline:
                break
            p = _random.randint(10**6, 10**9)
            # Make odd
            if p % 2 == 0:
                p += 1
            if is_prime(p):
                k = smooth_run_length(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p

    # Ensure at least k=1 (p^2+1 always... well, not always, but p=2: 5 is prime > 2)
    # Actually for p=2: p^2+1=5, largest factor 5 > 2, so k=0
    # For p=3: p^2+1=10=2*5, 5>3, so k=0
    # For p=5: p^2+1=26=2*13, 13>5, k=0
    # For p=7: p^2+1=50=2*5^2, all factors <=7, so k>=1
    #   p^2+2=51=3*17, 17>7, so k=1
    # For p=11: p^2+1=122=2*61, 61>11, k=0
    # For p=13: p^2+1=170=2*5*17, 17>13, k=0
    # For p=17: p^2+1=290=2*5*29, 29>17, k=0
    # For p=19: p^2+1=362=2*181, 181>19, k=0
    # For p=23: p^2+1=530=2*5*53, 53>23, k=0
    # For p=29: p^2+1=842=2*421, 421>29, k=0
    # For p=31: p^2+1=962=2*13*37, 37>31, k=0
    # For p=37: p^2+1=1370=2*5*137, 137>37, k=0
    # For p=41: p^2+1=1682=2*29^2, all factors <=41, k>=1
    #   p^2+2=1683=3*561=3*3*187=3^2*11*17, all <=41, k>=2
    #   p^2+3=1684=4*421=2^2*421, 421>41, k=2
    # So p=41 gives k=2
    
    # The seed found best k=2. Let's make sure we have at least that.
    if best_k < 1:
        # Fallback: find any valid (p, k) with k >= 1
        # p=7 gives k=1
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
