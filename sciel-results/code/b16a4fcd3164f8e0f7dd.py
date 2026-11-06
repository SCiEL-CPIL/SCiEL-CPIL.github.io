import time
import math
import sys
import random

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

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
    for a in _MR_BASES:
        if a % n == 0:
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

def sieve(n):
    if n < 2:
        return []
    is_prime_arr = bytearray(b'\x01') * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
    return [i for i in range(n+1) if is_prime_arr[i]]

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    y, c, m = random.randint(1, n-1), random.randint(1, n-1), random.randint(1, n-1)
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for i in range(r):
            y = (y*y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for i in range(min(m, r-k)):
                y = (y*y + c) % n
                q = (q * abs(x-y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (ys*ys + c) % n
            g = math.gcd(abs(x-ys), n)
            if g != 1:
                break
    return g

_SMALL_PRIMES_LIMIT = 100000
_small_primes = None

def _is_p_smooth_fast(n, p):
    """Check if all prime factors of n are <= p, optimized for n ~ p^2."""
    if n <= 1:
        return True
    if n <= p:
        return True
    # n > p, so n is not prime-smooth if n is prime
    # For n ~ p^2, if n is p-smooth, n must be composite with factors in [n/p, p]
    # Try to find a factor d such that n/p <= d <= p
    # First, quick trial division by very small primes
    global _small_primes
    rem = n
    # Only divide by primes up to min(p, 1000) for speed
    for q in _small_primes:
        if q > 1000 or q > p:
            break
        while rem % q == 0:
            rem //= q
    if rem == 1:
        return True
    if rem <= p:
        # rem is now <= p, but we need to check if rem is prime or all its factors <= p
        # Since rem <= p and we've divided out all factors <= 1000, if rem is prime it's <= p, OK
        # If rem is composite, its factors are > 1000 but rem <= p, so factors < rem <= p, OK
        return True
    # rem > p and rem > 1000
    # We need to check if rem has all factors <= p
    # rem is composite (if prime, it's > p, so not p-smooth)
    if is_prime(rem):
        return False
    # Use Pollard-Brent to factor rem
    # Check if all prime factors of rem are <= p
    def check_all_factors_leq(m, p):
        if m == 1:
            return True
        if m <= p:
            return True
        if is_prime(m):
            return m <= p
        d = pollard_brent(m)
        return check_all_factors_leq(d, p) and check_all_factors_leq(m // d, p)
    return check_all_factors_leq(rem, p)

def _check_window_segmented(p, k_max=200):
    """
    Use segmented sieve to quickly determine the maximum k such that
    all numbers p^2+1, ..., p^2+k have all prime factors <= p.
    
    Strategy:
    1. For each number n = p^2+i in the window, divide out all prime factors <= p.
    2. If any residual > 1 remains, that number has a prime factor > p, so k stops before i.
    
    We use a segmented approach: for each small prime q <= p, we mark multiples
    in the window [p^2+1, p^2+k_max] and divide them out.
    """
    global _small_primes
    base = p * p
    window_size = k_max
    
    # Initialize residuals for each number in the window
    # residuals[i] = p^2 + i with all small prime factors divided out
    residuals = [base + i for i in range(1, window_size + 1)]
    
    # For each prime q <= p, divide out q from all numbers in the window that are divisible by q
    for q in _small_primes:
        if q > p:
            break
        if q > 100000:  # Safety: don't iterate too many primes
            break
        
        # Find the first number in [base+1, base+window_size] divisible by q
        # base+1 to base+window_size
        # We need (base + i) % q == 0 for i in [1, window_size]
        # i.e., i ≡ (-base) mod q, and 1 <= i <= window_size
        
        first_i = (-base) % q
        if first_i == 0:
            first_i = q  # i must be at least 1, so if first_i==0, next is q
        # Actually: (-base) % q gives the remainder r such that base + r ≡ 0 (mod q)
        # We need i in [1, window_size] with i ≡ r (mod q)
        # If r == 0, then i = q, 2q, ...
        # If r > 0, then i = r, r+q, r+2q, ...
        
        r = (-base) % q
        if r == 0:
            start_i = q
        else:
            start_i = r
        
        # Now divide q out of residuals[start_i-1], residuals[start_i-1+q], etc.
        # (residuals is 0-indexed, residuals[i-1] corresponds to p^2+i)
        idx = start_i - 1
        while idx < window_size:
            # Divide q out of residuals[idx]
            while residuals[idx] % q == 0:
                residuals[idx] //= q
            idx += q
    
    # Now check each residual: if residuals[i-1] > 1, then p^2+i has a factor > p
    # (since we've divided out all primes <= p from the residual)
    # Wait, this is only true if we divided out ALL primes <= p, not just those <= 100000
    # For large p, we can't divide out all primes up to p efficiently.
    
    # Alternative: after dividing out small primes, check if residual is 1, <= p, or needs Pollard
    k = 0
    for i in range(1, window_size + 1):
        rem = residuals[i-1]
        if rem == 1:
            k = i
            continue
        if rem <= p:
            # rem <= p, so all its factors are <= p (since any factor of rem is <= rem <= p)
            k = i
            continue
        # rem > p: need to check if rem has all factors <= p
        # rem is not divisible by any prime <= 100000 (we divided those out)
        # If rem is prime, it's > p, so not p-smooth
        if is_prime(rem):
            break
        # rem is composite with all prime factors > 100000
        # Use Pollard-Brent to factor and check
        def check_all_factors_leq(m, p_limit):
            if m == 1:
                return True
            if m <= p_limit:
                return True
            if is_prime(m):
                return m <= p_limit
            d = pollard_brent(m)
            return check_all_factors_leq(d, p_limit) and check_all_factors_leq(m // d, p_limit)
        
        if not check_all_factors_leq(rem, p):
            break
        k = i
    
    return k

def construct():
    global _small_primes
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    if _small_primes is None:
        _small_primes = sieve(_SMALL_PRIMES_LIMIT)
        
    best_k = 0
    best_p = 2
    
    # Check small primes first using segmented sieve
    small_check_primes = sieve(20000)
    for p in small_check_primes:
        if time.time() > deadline - 5:
            break
        k = _check_window_segmented(p, 200)
        if k > best_k:
            best_k = k
            best_p = p

    # Random sampling with better prime candidate generation
    # Use a wheel to skip non-coprime-to-30 numbers
    wheel = [1, 7, 11, 13, 17, 19, 23, 29]
    while time.time() < deadline - 5:
        # Generate candidate using wheel
        base_num = random.randint(10**6, 10**9)
        offset = random.choice(wheel)
        p_candidate = base_num + offset
        if p_candidate < 10**6:
            p_candidate += 30
        if p_candidate > 10**9:
            p_candidate -= 30
        
        # Quick primality test
        if not is_prime(p_candidate):
            continue
            
        p = p_candidate
        k = _check_window_segmented(p, 200)
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
