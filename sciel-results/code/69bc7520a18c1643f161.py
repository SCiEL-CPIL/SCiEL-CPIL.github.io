"""Rewritten program: probabilistic pre-filtering + fast smoothness check."""
import time
import math
import random

# Deadline: evaluator kills after 150s; use 140s as hard limit with margin
DEADLINE = 140.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

# ---------- Deterministic Miller-Rabin ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
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
    for a in _MR_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = 128
        g = r = q = 1
        x = ys = y
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

def _factorize(n, limit=None):
    """Return list of prime factors of n. If limit is given, stop early if any factor > limit."""
    if n <= 1:
        return []
    if limit is not None and n > limit:
        # Quick check: if n is prime and > limit, it's a large factor
        # But we need to be careful; do trial division first
        pass
    factors = []
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        while n % p == 0:
            if limit is not None and p > limit:
                return None  # signal: has factor > limit
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    # Check if remaining n is prime
    if _is_prime(n):
        if limit is not None and n > limit:
            return None
        factors.append(n)
        return factors
    # Use Pollard-Brent to split
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_prime(m):
            if limit is not None and m > limit:
                return None
            factors.append(m)
            continue
        d = _pollard_brent(m)
        stack.append(d)
        stack.append(m // d)
    return factors

def _all_factors_le(n, limit):
    """Check if all prime factors of n are <= limit. Returns True/False."""
    if n <= limit:
        return True
    res = _factorize(n, limit)
    return res is not None

# ---------- Pre-filter: count how many of p^2+1..p^2+k are divisible by small primes ----------
# For a number to be p-smooth, it must be divisible only by primes <= p.
# A quick heuristic: for small primes q < p, p^2 + i ≡ i (mod q) if p ≡ 0 (mod q) which isn't the case.
# Actually p^2 + i mod q = (p mod q)^2 + i mod q.
# We want to check: does p^2 + i have a prime factor > p?
# A necessary condition for p^2+i to be p-smooth is that it is NOT coprime to the product of all primes <= p.
# But that's trivially true since p^2+i < p^2+200 and for i>=1, p^2+i is not divisible by p.
# Better heuristic: check if p^2+i is divisible by any small prime. If it has NO small prime factor,
# it's more likely to have a large prime factor.

_SMALL_PRIMES_FOR_FILTER = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _prefilter_score(p, k_max=200):
    """Score how likely p is to have a long smooth run.
    For each i in 1..k_max, check if p^2+i has at least one small prime factor.
    Higher score = more numbers have small factors = more likely smooth."""
    score = 0
    p_mod = {q: p % q for q in _SMALL_PRIMES_FOR_FILTER}
    p2_mod = {q: (p_mod[q] * p_mod[q]) % q for q in _SMALL_PRIMES_FOR_FILTER}
    for i in range(1, k_max + 1):
        has_small_factor = False
        for q in _SMALL_PRIMES_FOR_FILTER:
            if (p2_mod[q] + i) % q == 0:
                has_small_factor = True
                break
        if has_small_factor:
            score += 1
    return score

def _run_length(p):
    """Compute the actual run length k for prime p."""
    k = 0
    for i in range(1, 201):
        val = p * p + i
        if not _all_factors_le(val, p):
            break
        k += 1
    return k

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Search small primes exhaustively (up to some limit)
    # For small p, the run length can be checked quickly.
    # Phase 2: Use pre-filtering to rank larger primes, then check top candidates.
    
    # First, do a quick sweep of small primes
    p_limit_small = 50000
    for p in range(3, p_limit_small, 2):
        if _elapsed() > DEADLINE - 10:
            break
        if not _is_prime(p):
            continue
        k = _run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: For larger primes, use pre-filtering
    # We'll sample primes in ranges and rank them by prefilter score
    # Then check the top ones
    
    # Generate candidate primes in larger ranges
    # Use a sieve-like approach for primality testing of candidates
    # For p up to 10^9, we can't sieve all, so we'll sample
    
    # Strategy: for each range, test a batch of odd numbers for primality,
    # compute prefilter score, keep top candidates, then do full check.
    
    ranges = [
        (50000, 200000),
        (200000, 500000),
        (500000, 1000000),
        (1000000, 5000000),
        (5000000, 20000000),
        (20000000, 100000000),
        (100000000, 500000000),
        (500000000, 1000000000),
    ]
    
    for lo, hi in ranges:
        if _elapsed() > DEADLINE - 15:
            break
        # Sample step: for large ranges, don't test every odd number
        # Estimate how many we can test in remaining time
        remaining = DEADLINE - 15 - _elapsed()
        if remaining <= 0:
            break
        
        # For each range, test a subset
        # The prefilter is cheap; full check is expensive.
        # Let's test up to ~5000 candidates per range, prioritized by prefilter.
        
        # Collect candidates with their prefilter scores
        candidates = []
        step = max(2, (hi - lo) // 50000)  # sample ~50000 points, then filter
        # Actually, let's be smarter: test every odd number in a subset
        
        # For efficiency, test a random-ish subset of odd numbers
        # But deterministic is better for reproducibility. Let's use a stride.
        
        # Number of odd numbers in range
        n_odds = (hi - lo) // 2 + 1
        # We'll test at most ~10000 candidates per range
        max_test = 10000
        if n_odds <= max_test:
            stride = 1
        else:
            stride = max(1, n_odds // max_test)
        
        count_tested = 0
        for start in range(lo + (1 if lo % 2 == 0 else 0), hi + 1, 2 * stride):
            if _elapsed() > DEADLINE - 15:
                break
            if start > hi:
                break
            if not _is_prime(start):
                continue
            count_tested += 1
            score = _prefilter_score(start)
            candidates.append((score, start))
        
        # Sort by score descending, check top candidates
        candidates.sort(reverse=True)
        # Check top candidates, limited by time
        for score, p in candidates:
            if _elapsed() > DEADLINE - 5:
                break
            if score < best_k:  # even if all pass prefilter, can't beat best
                # Actually score is out of 200; if score < best_k, max possible k <= score
                # But score is just count of numbers with small factors, not a tight upper bound
                # Still, it's a useful heuristic
                pass
            k = _run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
            # Early stop: if we've checked several and none improve, maybe stop
            # But let's check a few more
        
        if _elapsed() > DEADLINE - 5:
            break
    
    return {'p': best_p, 'k': best_k}
