"""Rewritten program using segmented sieve for p-smoothness verification."""
import math

def _is_prime(n):
    """Deterministic Miller-Rabin for n < 2^64 (more than enough for our range)."""
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Bases sufficient for n < 2^64
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
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

def _pollard_brent(n):
    """Pollard's Brent cycle-finding factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = 2, 1, 128
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
    if g == n:
        # Retry with different parameters
        return _pollard_brent(n)
    return g

def _factorize(n):
    """Fully factorize n using trial division for small primes and Pollard-Brent."""
    factors = []
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    if _is_prime(n):
        factors.append(n)
        return factors
    # Use Pollard-Brent for remaining composite
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_prime(m):
            factors.append(m)
        else:
            d = _pollard_brent(m)
            if d == m:
                # Should not happen often; try again
                d = _pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
    return factors

def _max_prime_factor(n):
    """Return the largest prime factor of n."""
    if n <= 1:
        return 1
    factors = _factorize(n)
    return max(factors)

def construct():
    """
    Find a prime p <= 10^9 and k <= 200 such that all of p^2+1, ..., p^2+k
    have all prime factors <= p. Maximize k.
    
    Strategy:
    - For each candidate prime p, check the longest run starting at p^2+1
      where each number is p-smooth (all prime factors <= p).
    - Use efficient factorization with trial division + Pollard-Brent.
    - Search over primes in increasing order, but focus on ranges likely to
      yield longer runs.
    """
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Quick check: if n has a prime factor > p, return False
        # First try small primes up to some limit
        # Then use full factorization
        factors = _factorize(n)
        return all(f <= p for f in factors)
    
    # We'll search over primes. The key insight is that for small p,
    # the numbers p^2+1 through p^2+k are relatively small and may have
    # large prime factors. For larger p, there's more "room" for smoothness.
    
    # Let's try a range of primes. Based on the seed, it searched up to 2000.
    # Let's extend the search to larger primes but be smart about it.
    
    # First, let's do a quick scan over small primes to find a good starting point
    best_k = 0
    best_p = None
    
    # Generate primes up to a reasonable limit for the search
    # We'll test primes in chunks, focusing on where runs might be longer
    
    # For efficiency, let's test primes up to around 100000 first
    # Then if time permits, go higher
    
    def check_prime(p):
        """Check the longest run of p-smooth numbers starting at p^2+1."""
        k = 0
        base = p * p
        while k < 200:
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        return k
    
    # Search over primes
    # Let's be strategic: test primes in ranges
    # Small primes: 3 to 1000
    # Medium primes: 1000 to 10000
    # Large primes: 10000 to 100000
    
    # Generate candidate primes
    def gen_primes(limit):
        """Simple sieve for primes up to limit."""
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    # Phase 1: Small primes (3 to 5000)
    small_primes = gen_primes(5000)
    for p in small_primes:
        if p < 3:
            continue
        k = check_prime(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Medium primes (5000 to 50000) - sample some
    # For larger primes, checking all 200 numbers is more expensive
    # Let's check a subset
    medium_primes = gen_primes(50000)
    # Filter to those > 5000
    medium_primes = [p for p in medium_primes if p > 5000]
    # Sample every 10th prime to save time
    for p in medium_primes[::10]:
        k = check_prime(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 3: If best_k is still small, try some larger primes
    if best_k < 10:
        large_primes = gen_primes(100000)
        large_primes = [p for p in large_primes if p > 50000]
        for p in large_primes[::50]:
            k = check_prime(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    if best_p is None:
        best_p = 3
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
