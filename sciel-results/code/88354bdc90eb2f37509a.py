import time
import random
from bisect import bisect_right

_START = time.time()
_DEADLINE = _START + 118.0

def _time_ok():
    return time.time() < _DEADLINE

# --- Deterministic Miller-Rabin ---
def is_prime(n):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases for 64-bit
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
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

# --- Pollard-Brent ---
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def pollard_brent(n, rng):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = rng.randrange(2, n - 1)
        c = rng.randrange(1, n - 1)
        m = rng.randrange(1, 100)
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
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

_rng = random.Random(12345)

def largest_prime_factor(n):
    if n < 2:
        return 1
    if is_prime(n):
        return n
    f = pollard_brent(n, _rng)
    return max(largest_prime_factor(f), largest_prime_factor(n // f))

def is_p_smooth(n, p):
    if n < 2:
        return True
    # Quick check: if n <= p, it's trivially p-smooth? No, factors must be <= p.
    # If n is prime and n <= p, True.
    # If n is composite, factors are < n <= p. So True.
    if n <= p:
        return True
    # If n > p, we need to check if all prime factors are <= p.
    # This is equivalent to largest_prime_factor(n) <= p.
    return largest_prime_factor(n) <= p

# --- Sieve ---
def _sieve(n):
    bs = bytearray([1]) * (n + 1)
    bs[0:2] = b'\0\0'
    for i in range(2, int(n ** 0.5) + 1):
        if bs[i]:
            bs[i * i::i] = b'\0' * len(bs[i * i::i])
    return [i for i in range(2, n + 1) if bs[i]]

# Generate primes up to 10^6 for candidate p
_PRIMES_UP_TO = 1000000
_PRIMES = _sieve(_PRIMES_UP_TO)

def construct():
    global _rng
    rng = random.Random(98765)
    _rng = rng
    
    best_k = 0
    best_p = None
    
    # We search primes p from the list _PRIMES.
    # We want to test primes that are large enough to have a chance of long runs,
    # but not too large to be computationally expensive.
    # p around 10^4 to 10^6.
    
    # Filter primes to a subset for testing to manage time.
    # Test primes in [1000, 100000] first.
    # Then [100000, 1000000] if time permits.
    
    # To maximize k, we should test many p.
    # The probability of a run of length k is roughly rho(2)^k.
    # rho(2) ~ 0.3.
    # k=10 -> 0.3^10 ~ 6e-6.
    # We need ~ 1e5 candidates to find one.
    
    # We will iterate through primes in _PRIMES.
    # Skip very small primes (< 100) as p^2+i is small and likely has large factors relative to p?
    # Actually, for small p, p-smooth is easier?
    # If p=10, p^2+1=101 (prime > 10). Fail.
    # If p=100, p^2+1=10001=73*137. 137>100. Fail.
    # It seems small p fail quickly.
    
    # Let's iterate all primes in _PRIMES that are >= 100.
    # But we must be mindful of time.
    # is_p_smooth uses largest_prime_factor which uses Pollard-Brent.
    # Pollard-Brent is fast for 64-bit numbers.
    # n ~ p^2 ~ 10^12.
    
    # Strategy:
    # 1. Iterate primes p.
    # 2. For each p, check i=1, 2, ...
    # 3. Stop if is_p_smooth fails.
    # 4. Update best_k.
    # 5. If time runs out, break.
    
    # To improve search, we can skip p that are "bad" quickly.
    # Most p will fail at i=1 or i=2.
    
    # Let's do a random sampling if systematic is too slow?
    # Systematic is better for coverage.
    
    # We'll iterate _PRIMES.
    # To save time, we can test a subset first.
    # Let's test primes in steps.
    
    # Optimization: Pre-compute p^2.
    
    for p in _PRIMES:
        if not _time_ok():
            break
        if p < 100:
            continue
        # If p is too large, p^2 might be > 10^18? 
        # p <= 10^6 -> p^2 <= 10^12. Fits in 64-bit int.
        
        k = 0
        base = p * p
        # Check up to 200 or until fail
        while k < 200:
            if not _time_ok():
                break
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # Optional: if we find a very good k, we can stop or adjust search.
            # But we want to maximize, so keep going.
            
    if best_p is None:
        # Fallback
        best_p = 7
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
