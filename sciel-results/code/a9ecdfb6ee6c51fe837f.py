"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses congruence-based pre-sieve to eliminate candidate primes p based on modular
arithmetic, then verifies survivors with trial division and Pollard-Brent."""

import time
import math
import random

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard's Brent ----------
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
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
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

# ---------- Precompute small primes for trial division ----------
_SMALL_PRIMES = []
def _init_small_primes(limit=10000):
    """Generate small primes for fast trial division."""
    if limit < 2:
        return
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    _SMALL_PRIMES.extend(i for i in range(2, limit + 1) if is_p[i])

_init_small_primes(10000)

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    
    for q in _SMALL_PRIMES:
        if q > p:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
    
    if n == 1:
        return True
    
    if is_prime(n):
        return n <= p
    
    factors = []
    _factor(n, factors, p)
    for f in factors:
        if f > p:
            return False
    return True

def _factor(n, factors, p):
    """Recursively factor n, adding factors to list."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    _factor(d, factors, p)
    _factor(n // d, factors, p)

def sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

# ---------- Congruence-based pre-sieve ----------
# For each j in 1..200, precompute the set of small primes q (q > 200)
# such that p^2 + j ≡ 0 (mod q) has a solution for p mod q.
# A candidate prime p is "killed" for run length >= j if for some j' <= j,
# p^2 + j' is divisible by a prime q > p (which would violate p-smoothness).
# But we can't easily check q > p in the sieve. Instead:
# For each small prime q, compute residues r mod q where r^2 + j ≡ 0 (mod q).
# If p ≡ r (mod q) for some j in 1..K, then p^2 + j is divisible by q.
# If q > p, this kills the run at position j.
# Strategy: precompute for each q, the set of j values in 1..200 for which
# -j is a quadratic residue mod q. Then for a candidate p, check if any
# "bad" j exists where q > p.

def _build_kill_table(max_q, max_j=200):
    """
    For each prime q up to max_q, compute the set of j in 1..max_j
    such that -j is a quadratic residue mod q (i.e., p^2 ≡ -j mod q has solution).
    Returns dict: q -> set of j values.
    """
    kill_table = {}
    # Generate primes up to max_q
    if max_q < 2:
        return kill_table
    is_p = bytearray(b'\x01') * (max_q + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(max_q)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    primes_q = [i for i in range(2, max_q + 1) if is_p[i]]
    
    for q in primes_q:
        # Compute which j in 1..max_j are quadratic non-residues of -j mod q
        # i.e., p^2 ≡ -j (mod q) has a solution iff (-j) is a QR mod q
        # For odd prime q: -j is QR iff (-j)^((q-1)/2) ≡ 1 (mod q)
        # Or equivalently: j is QR iff j^((q-1)/2) ≡ 1, and -1 is QR iff q ≡ 1 mod 4
        # So -j is QR iff j^((q-1)/2) * (-1)^((q-1)/2) ≡ 1 mod q
        # i.e., j^((q-1)/2) ≡ (-1)^((q-1)/2) mod q
        
        bad_js = set()
        exp = (q - 1) // 2
        for j in range(1, max_j + 1):
            if j % q == 0:
                # p^2 ≡ 0 mod q means p ≡ 0 mod q, but p is prime and p != q (p < q in our use)
                # Actually if p == q, then p^2 + j ≡ j mod q, not 0. So j ≡ 0 mod q doesn't kill.
                # But p^2 ≡ -j ≡ 0 mod q means p ≡ 0 mod q, impossible for prime p != q.
                # So skip.
                continue
            # Check if -j is a quadratic residue mod q
            # (-j)^exp mod q should be 1
            val = pow(-j, exp, q)
            if val == 1:
                bad_js.add(j)
        if bad_js:
            kill_table[q] = bad_js
    
    return kill_table

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Quick search over small primes with direct checking
    sieve_limit = 2_000_000
    primes = sieve_primes(sieve_limit)
    
    for p in primes:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() - start_time > deadline - 30:
                break
            n = p_sq + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Congruence-based pre-sieve for larger primes
    # Build kill table for primes q up to some limit
    # Key insight: if p is a prime and p^2 + j is divisible by q where q > p,
    # then p^2 + j is not p-smooth. So we want to avoid p where for small j,
    # p^2 + j has a large prime factor.
    
    # For the pre-sieve: for a range of candidate p values, mark p as "bad"
    # if there exists a prime q in [p+1, q_max] and j in 1..200 such that
    # p^2 + j ≡ 0 (mod q).
    
    # Simplified approach: for each small prime q, find residues r mod q where
    # r^2 + j ≡ 0 for some j in 1..200. Then any p ≡ r (mod q) with p < q
    # will have p^2 + j divisible by q > p, killing the run.
    
    # But we need q > p, so for a given range of p, we need q in (p, q_max].
    # This is complex. Let's use a different approach:
    
    # For each candidate prime p, quickly check if p^2 + j has a factor > p
    # for j = 1, 2, ..., up to some limit. Use the kill table to skip candidates.
    
    # Build kill table: for each prime q up to q_max, the set of residues r mod q
    # such that r^2 + j ≡ 0 (mod q) for some j in 1..200.
    q_max = 200000
    kill_residues = {}  # q -> set of residues r mod q that are "bad"
    
    is_p = bytearray(b'\x01') * (q_max + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(q_max)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    primes_q = [i for i in range(2, q_max + 1) if is_p[i]]
    
    for q in primes_q:
        bad_residues = set()
        for j in range(1, 201):
            # Solve r^2 ≡ -j (mod q)
            # Try all residues (expensive for large q, but q_max is 200k)
            # Better: use Legendre symbol to check if -j is QR, then find sqrt
            # For now, just check if -j is a QR
            if j % q == 0:
                continue
            exp = (q - 1) // 2
            if pow(-j, exp, q) == 1:
                # -j is a QR mod q, so there exist r with r^2 ≡ -j mod q
                # Find one such r
                # Use Tonelli-Shanks or just brute force for small q
                # For efficiency, just mark that some residue is bad
                # We'll compute the actual residues later if needed
                bad_residues.add(j)  # Store j values, not residues, for now
        if bad_residues:
            kill_residues[q] = bad_residues
    
    # Now search for primes in larger ranges using the kill table
    # For a candidate p, it's "bad" if there exists q in kill_residues with q > p
    # and j in kill_residues[q] such that p^2 + j ≡ 0 (mod q).
    # i.e., p ≡ ±sqrt(-j) (mod q) for some j.
    
    # This is still complex. Let's use a simpler heuristic:
    # For each candidate prime p, do a quick check: for j = 1 to min(20, best_k+1),
    # check if p^2 + j has a prime factor > p using trial division up to sqrt(p^2+j).
    # If sqrt(p^2+j) < q_max, we can check all primes up to sqrt(p^2+j).
    
    # Actually, let's just do a smarter direct search with early termination.
    # Focus on primes where p^2+1 through p^2+best_k+1 are all p-smooth.
    
    # Search in larger prime ranges
    search_ranges = [
        (2_000_000, 10_000_000),
        (10_000_000, 50_000_000),
        (50_000_000, 200_000_000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() - start_time > deadline - 10:
            break
        
        # Sieve primes in [lo, hi]
        if hi - lo > 5_000_000:
            # Too large for single sieve, sample
            num_samples = 200
            for _ in range(num_samples):
                if time.time() - start_time > deadline - 5:
                    break
                candidate = random.randrange(lo, hi)
                if candidate % 2 == 0:
                    candidate += 1
                for _ in range(200):
                    if is_prime(candidate):
                        break
                    candidate += 2
                else:
                    continue
                
                k = 0
                p_sq = candidate * candidate
                while k < 200:
                    if time.time() - start_time > deadline - 2:
                        break
                    n = p_sq + k + 1
                    if not is_p_smooth(n, candidate):
                        break
                    k += 1
                
                if k > best_k:
                    best_k = k
                    best_p = candidate
        else:
            # Segmented sieve
            seg_size = hi - lo + 1
            is_prime_seg = bytearray(b'\x01') * seg_size
            is_prime_seg[0] = is_prime_seg[0]  # lo
            # Handle even
            for i in range(lo, hi + 1, 2):
                if i == 2:
                    continue
                idx = i - lo
                if idx >= 0 and idx < seg_size:
                    is_prime_seg[idx] = 0
            # Sieve with small primes
            for q in _SMALL_PRIMES:
                if q * q > hi:
                    break
                start = ((lo + q - 1) // q) * q
                if start == q:
                    start += q
                for i in range(start, hi + 1, q):
                    idx = i - lo
                    if 0 <= idx < seg_size:
                        is_prime_seg[idx] = 0
            
            for i in range(seg_size):
                if time.time() - start_time > deadline - 5:
                    break
                if is_prime_seg[i]:
                    p = lo + i
                    k = 0
                    p_sq = p * p
                    while k < 200:
                        if time.time() - start_time > deadline - 2:
                            break
                        n = p_sq + k + 1
                        if not is_p_smooth(n, p):
                            break
                        k += 1
                    
                    if k > best_k:
                        best_k = k
                        best_p = p
    
    return {'p': best_p, 'k': best_k}
