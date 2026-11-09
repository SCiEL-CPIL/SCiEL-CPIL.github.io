import time
import math
import random
import numpy as np

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MILLER_RABIN_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MILLER_RABIN_BASES:
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

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Segmented Sieve ----------
def _sieve_simple(limit):
    """Simple sieve for small limits (up to ~10^6)"""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def _segmented_sieve(limit):
    """
    Generate all primes up to limit using segmented sieve.
    Returns a list of primes.
    """
    if limit < 2:
        return []
    
    # First, get primes up to sqrt(limit) using simple sieve
    sqrt_limit = int(math.isqrt(limit)) + 1
    small_primes = _sieve_simple(sqrt_limit)
    
    all_primes = list(small_primes)
    
    if limit <= sqrt_limit:
        return all_primes
    
    # Segmented sieve for the rest
    seg_size = 1 << 20  # 1M segment size
    low = sqrt_limit + 1
    # Align to even numbers for efficiency (skip even numbers)
    # Actually let's just do standard segmented sieve
    
    while low <= limit:
        high = min(low + seg_size - 1, limit)
        seg = bytearray(b'\x01') * (high - low + 1)
        
        for p in small_primes:
            if p * p > high:
                break
            # Find first multiple of p in [low, high]
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, high + 1, p):
                seg[j - low] = 0
        
        for i in range(high - low + 1):
            if seg[i]:
                all_primes.append(low + i)
        
        low = high + 1
    
    return all_primes

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Phase 1: Generate primes up to 10^7 using segmented sieve
    SIEVE_LIMIT = 10_000_000
    primes = _segmented_sieve(SIEVE_LIMIT)
    
    # Also keep small primes for trial division (primes up to 10^5)
    small_primes = [p for p in primes if p <= 100_000]
    
    best_k = 0
    best_p = 3
    
    # Phase 2: For each prime p, check the window [p^2+1, p^2+200]
    # Strategy: 
    #   1. For small p (p < 10^5), use trial division by all primes <= p
    #   2. For larger p, use vectorized NumPy filter for primes <= p, 
    #      then Pollard-Brent for remaining large factors
    
    # Precompute primes as numpy array for vectorized operations
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Process primes in order of increasing p
    # For each p, we need all prime factors of p^2+i (i=1..200) to be <= p
    
    # Key insight: For p >= 10^5, p^2 is huge (~10^10), and the values p^2+i 
    # have at most ~2-3 prime factors. The condition "all factors <= p" is 
    # equivalent to: after removing all prime factors <= p, the remainder is 1.
    # Since p^2+i < p^2+p for i<=p, any prime factor > p must be > sqrt(p^2+i) 
    # only if it's the largest factor... Actually for p^2+i, if it has a factor 
    # > p, that factor must be < p^2+i/p ~ p + i/p, so it's close to p.
    
    # More precisely: if p^2+i = a*b with a <= b, and a > p, then b < (p^2+i)/p = p + i/p < p+1
    # So both a and b would be in (p, p+1), impossible for integers.
    # Wait: a > p and b >= a > p, so a*b > p^2. But p^2+i > p^2.
    # If a > p, then b = (p^2+i)/a < (p^2+i)/p = p + i/p.
    # So b < p + 1 (since i <= 200 and p >= 1, i/p < 1 for p > 200).
    # For p > 200: if p^2+i has a factor a > p, then the other factor b < p+1, 
    # so b <= p. Then a = (p^2+i)/b > p^2/p = p. 
    # So p^2+i = b * a where b <= p and a > p.
    # The condition "all prime factors <= p" means a must also be <= p, 
    # but a > p, contradiction. So for p > 200, p^2+i is p-smooth 
    # iff after dividing out all prime factors <= p, the remainder is 1.
    
    # Actually simpler: p^2+i is p-smooth iff it has no prime factor > p.
    # Since p^2+i < (p+1)^2 for i < 2p+1 (which is true for i<=200, p>=100),
    # p^2+i can have at most one prime factor > p (because if it had two, 
    # their product > p^2, and the third factor would make it > p^3 > p^2+200 for large p).
    # Actually: if p^2+i = q1*q2*... where all qi > p, then product > p^k.
    # For k=2: q1*q2 > p^2. But q1*q2 = p^2+i, so this is possible.
    # For k=3: q1*q2*q3 > p^3 >> p^2+200. So at most 2 prime factors > p.
    # But if q1 > p and q2 > p, then q1*q2 > p^2, and q1*q2 divides p^2+i,
    # so q1*q2 <= p^2+i. Thus p^2 < q1*q2 <= p^2+i, so i >= q1*q2 - p^2.
    # This is a very tight constraint.
    
    # Practical approach: For each p, compute the "p-smoothness" by:
    # 1. Trial divide by primes <= min(p, sqrt(p^2+200)) = min(p, p+something) ~ p
    #    Actually sqrt(p^2+200) ~ p, so trial divide by all primes <= p.
    # 2. If remainder is 1, it's p-smooth.
    # 3. If remainder > 1, check if it's prime and <= p (for the case where 
    #    the remainder is a single prime factor).
    # 4. Otherwise use Pollard-Brent.
    
    # For efficiency with large p:
    # - Trial division by all primes <= p is expensive for p ~ 10^7 (there are ~664k primes)
    # - But p^2+i has at most ~4 prime factors total, so most trial divisions will fail quickly
    # - Better: use the fact that we only need to check if p^2+i is p-smooth
    
    # OPTIMIZATION: For a given p, p^2+i is p-smooth iff:
    #   Let m = p^2+i. Remove all prime factors q <= p from m.
    #   If result is 1, m is p-smooth.
    #   If result is a prime r, m is p-smooth iff r <= p.
    #   If result is composite, factor it and check all factors <= p.
    
    # Since m ~ p^2, and we're dividing by primes up to p, the number of 
    # trial divisions needed is at most the number of distinct prime factors 
    # of m that are <= p, which is small (usually 2-4).
    
    # But we still need to iterate through primes to find which ones divide m.
    # For large p, most primes won't divide m, so we're doing ~664k mod operations per m.
    # With 200 values per p, that's ~133M mod ops per p. Too slow.
    
    # BETTER APPROACH: Precompute for each m = p^2+i, factor it using Pollard-Brent
    # directly, then check if all factors <= p. This avoids the trial division loop.
    
    # Even better: For the specific structure m = p^2 + i, we can note:
    # - m is odd (if p is odd, p^2 is odd, so p^2+i is odd when i is even, even when i is odd)
    # - We can use Pollard-Brent which is fast for semiprimes and numbers with small factors
    
    # Let's use a hybrid: 
    # - For p <= 10^5: trial division by primes <= p (feasible, ~9592 primes)
    # - For p > 10^5: direct Pollard-Brent factorization of each m, check factors <= p
    
    # Actually, let's benchmark: for p ~ 10^6, p^2 ~ 10^12. 
    # Pollard-Brent on 10^12 numbers is fast (microseconds).
    # Trial division by 78k primes is slower.
    # So for p > 10^5, use Pollard-Brent directly.
    
    # For p <= 10^5, trial division is fine.
    
    small_prime_limit = 100_000
    small_primes_list = [p for p in primes if p <= small_prime_limit]
    
    # Process all primes
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
        
        p_sq = p * p
        
        # Quick check: can we achieve a longer run than current best?
        # If we've already found best_k, we need to check at least best_k+1 values
        # to beat it. But we still need to check all 200 to maximize.
        
        k = 0
        max_check = 200
        
        if p <= small_prime_limit:
            # Trial division approach
            for i in range(1, max_check + 1):
                if time.time() > deadline:
                    break
                val = p_sq + i
                t = val
                for q in small_primes_list:
                    if q > p:
                        break
                    if q * q > t:
                        break
                    while t % q == 0:
                        t //= q
                if t == 1:
                    k = i
                    continue
                # t > 1: check if t is prime and <= p
                if _is_prime_mr(t):
                    if t <= p:
                        k = i
                    else:
                        break
                else:
                    # Composite remainder, factor it
                    factors = []
                    _factor(t, factors)
                    if all(f <= p for f in factors):
                        k = i
                    else:
                        break
        else:
            # Pollard-Brent approach for each value
            for i in range(1, max_check + 1):
                if time.time() > deadline:
                    break
                val = p_sq + i
                # Factor val completely
                factors = []
                _factor(val, factors)
                if all(f <= p for f in factors):
                    k = i
                else:
                    break
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break
    
    # Phase 3: If we haven't found a great k, try larger primes (10^7 to 10^9)
    # by sampling. For very large p, the condition is easier to satisfy.
    if best_k < 150:
        # Sample primes in [10^7, 10^9]
        # For p ~ 10^8, p^2 ~ 10^16. Pollard-Brent handles this well.
        # The key insight: for large p, p^2+i is likely to be p-smooth 
        # because any large prime factor would need to be > p, but p^2+i 
        # has limited "room" for such factors.
        
        attempts = 0
        max_attempts = 50000
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Random odd number in [10^7, 10^9]
            lo = 10_000_000
            hi = 1_000_000_000
            p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            
            # Check if prime
            if not _is_prime_mr(p):
                continue
            
            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + i
                factors = []
                _factor(val, factors)
                if all(f <= p for f in factors):
                    k = i
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break
    
    return {'p': best_p, 'k': best_k}
