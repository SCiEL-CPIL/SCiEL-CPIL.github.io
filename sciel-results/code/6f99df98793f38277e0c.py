"""Long runs after p^2 with no prime factor exceeding p.

Strategy:
  1. Deterministic Miller-Rabin for primality.
  2. Pollard-Brent for factoring.
  3. Segmented-sieve-style precomputation: for small primes q, determine which
     residue classes mod q of p make p^2 + i divisible by q for some i in [1,200].
     Use this to rapidly reject most candidate primes p without full factorization.
  4. For surviving candidates, verify the full condition with exact factorization.
  5. Deadline guard to stay within 120 s.
"""
import math
import random
import time

# ---------- Deterministic Miller-Rabin ----------

def _miller_rabin(n):
    """Deterministic Miller-Rabin primality test for n < 3.3e24."""
    if n < 2:
        return False
    if n == 2 or n == 3 or n == 5 or n == 7 or n == 11 or n == 13 or n == 17 or n == 19 or n == 23 or n == 29 or n == 31 or n == 37:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases sufficient for n < 3.3e24
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_prime(n):
    return _miller_rabin(n)


# ---------- Pollard-Brent ----------

def pollard_brent(n):
    """Return a non-trivial factor of n using Pollard-Brent."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    if n % 7 == 0:
        return 7
    if n % 11 == 0:
        return 11
    if n % 13 == 0:
        return 13
    if n % 17 == 0:
        return 17
    if n % 19 == 0:
        return 19
    if n % 23 == 0:
        return 23
    if n % 29 == 0:
        return 29
    if n % 31 == 0:
        return 31
    if n % 37 == 0:
        return 37
    if is_prime(n):
        return n
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
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
        # retry


def factorize(n):
    """Full factorization of n into prime factors (with multiplicity)."""
    if n <= 1:
        return []
    if is_prime(n):
        return [n]
    factors = []
    # trial division by small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors.append(m)
            continue
        d = pollard_brent(m)
        stack.append(d)
        stack.append(m // d)
    return factors


def largest_prime_factor(n):
    if n <= 1:
        return 1
    f = factorize(n)
    return max(f) if f else 1


# ---------- Precomputation: forbidden residues ----------

def precompute_forbidden(max_p=10**9, max_offset=200, max_q=10**6):
    """
    For each small prime q <= max_q, compute the set of residue classes r mod q
    such that there exists i in [1, max_offset] with r^2 ≡ -i (mod q).
    If p ≡ r (mod q) for such r, then p^2 + i ≡ 0 (mod q) for some i, meaning
    p^2 + i has q as a factor. Since q <= max_q and we need all factors <= p,
    this is only a problem if q > p. But for small p, q could exceed p.
    
    Actually the condition is: every prime factor of p^2+i must be <= p.
    If q divides p^2+i and q > p, that's a violation.
    So for a given p, we need: for all i in [1,k], no prime factor of p^2+i exceeds p.
    
    For the sieve: if q is a small prime and q > p, then q dividing p^2+i is bad.
    But q > p means q is large relative to p. For p up to 10^9, small q <= 10^6 
    are all < p for p > 10^6. So for p > 10^6, small-q divisibility is NOT a violation.
    
    Reconsider: the condition is violated only if some prime factor of p^2+i is > p.
    For p > 10^6, all primes q <= 10^6 are < p, so they don't cause violations.
    The sieve on small q doesn't help for large p.
    
    Better approach: just iterate over candidate primes p and check the condition.
    """
    return None


# ---------- Main construct ----------

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    best_k = 0
    best_p = 2

    # Check small primes first (they're fast)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def check_prime(p):
        """Check how long the run is for prime p. Returns k."""
        k = 0
        for i in range(1, 201):
            if time.time() > deadline - 10:
                break
            n = p * p + i
            lp = largest_prime_factor(n)
            if lp <= p:
                k += 1
            else:
                break
        return k

    # Try small primes
    for p in small_primes:
        if time.time() > deadline - 5:
            break
        k = check_prime(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Now search larger primes using a segmented sieve approach
    # We'll generate candidate primes in ranges and test them.
    
    def check_prime_fast(p):
        """Faster check: trial divide by primes up to p first."""
        k = 0
        for i in range(1, 201):
            if time.time() > deadline - 10:
                break
            n = p * p + i
            # Trial divide by primes up to sqrt(n), but we only care if any factor > p
            # So: factor n, check if any prime factor > p
            rem = n
            # Trial divide by small primes up to p (but stop early if factor > p found)
            # Actually, if any prime factor > p, we fail.
            # So we need to check: does n have a prime factor > p?
            # Equivalent: after removing all prime factors <= p, is remainder > 1?
            # But we can short-circuit: trial divide by primes <= p; if at any point
            # the remaining n is > 1 and is_prime(remaining) and remaining > p, fail.
            # Simpler: full factorize and check max.
            # For speed, trial divide up to p, then check if remainder is 1 or prime <= p.
            d = 2
            limit = p
            while d * d <= rem and d <= limit:
                if rem % d == 0:
                    # d <= p, ok, remove
                    while rem % d == 0:
                        rem //= d
                d += 1 if d == 2 else 2
            # After removing all factors <= p (and up to sqrt(rem)):
            if rem > 1:
                # rem must be prime (since we divided up to sqrt) or composite with both factors > p
                if rem > p:
                    return k  # failure
                # else rem <= p, it's fine (it's a prime <= p)
            k += 1
        return k

    # Generate primes in segments using a simple sieve
    # Start from 100 and go up, testing primes
    
    def is_prime_trial(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        if n == 3:
            return True
        if n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # Use Miller-Rabin for primality testing in the search
    def is_prime_fast(n):
        if n < 2:
            return False
        if n == 2 or n == 3 or n == 5 or n == 7 or n == 11 or n == 13:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        return _miller_rabin(n)

    # Segmented search: test numbers in blocks, filter by primality, then check
    # Start from 101 (after small primes)
    lower = 101
    # We'll go up to some limit, but constrained by time
    # For each candidate prime p, checking is the bottleneck.
    
    # Optimize: for a given p, the worst case is checking all 200 values.
    # Each check involves factorization. Let's estimate: for p ~ 10^6, p^2 ~ 10^12.
    # Factorization of 10^12 numbers is expensive.
    
    # Better strategy: iterate p in increasing order, and for each, compute the run length.
    # Stop when we've exhausted the time budget.
    
    # Let's also try a targeted approach: for each i in [1,200], we want p^2 + i to be
    # p-smooth. This is related to smooth numbers.
    
    # For now, do a systematic search with time management.
    search_start = 101
    while search_start < 10**9 and time.time() < deadline - 20:
        # Check if search_start is prime
        if is_prime_fast(search_start):
            k = check_prime_fast(search_start)
            if k > best_k:
                best_k = k
                best_p = search_start
                if best_k >= 200:
                    break
        search_start += 2  # only odd numbers (except 2 already handled)
        
        # Skip even numbers
        if search_start % 2 == 0:
            search_start += 1

    return {'p': best_p, 'k': best_k}
