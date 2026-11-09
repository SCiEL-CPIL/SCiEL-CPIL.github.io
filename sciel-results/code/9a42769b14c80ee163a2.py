import time
import math
import random
import sys

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
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

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Sieve ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def _is_p_smooth(n, p, small_primes):
    """
    Check if n is p-smooth (all prime factors <= p).
    Uses trial division by small_primes, then Pollard-Brent for remainder.
    """
    t = n
    # Trial division by small primes up to p
    for q in small_primes:
        if q > p:
            break
        if q * q > t:
            break
        while t % q == 0:
            t //= q
    
    if t == 1:
        return True
    
    # If t is prime, check if it's <= p
    if _is_prime_mr(t):
        return t <= p
    
    # t is composite. Factor it.
    # Note: t has no factors <= min(p, max(small_primes))
    # If p is large, t might still have small factors we didn't check?
    # No, we checked all small_primes up to p.
    # If p > max(small_primes), we didn't check primes between max(small_primes) and p.
    # But if t has a factor q in (max(small_primes), p], we missed it.
    # However, if t is composite and has a factor q <= p, we need to find it.
    # Pollard-Brent will find *a* factor.
    factors = []
    _factor(t, factors)
    return all(f <= p for f in factors)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute small primes up to 100000 for fast trial division
    SIEVE_LIMIT = 100000
    small_primes = _sieve(SIEVE_LIMIT)
    
    best_k = 0
    best_p = 3
    
    # Phase 1: Search small primes (up to 100000)
    # This is fast and might find a good local maximum early.
    for p in small_primes:
        if time.time() > deadline:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + i
            if _is_p_smooth(val, p, small_primes):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                return {'p': best_p, 'k': best_k}
                
    # Phase 2: Random search for larger primes
    # The space of valid p is sparse. Random sampling of large primes
    # allows us to find rare instances with longer runs.
    random.seed(42)
    
    # We search for primes in [100000, 10^9]
    # Generating primes randomly:
    # Pick random odd x, check primality.
    
    # To make this efficient, we can skip even numbers.
    # Also, we can use a small sieve to filter out numbers divisible by small primes
    # before doing Miller-Rabin.
    
    # Precompute a set of small primes for quick divisibility check
    filter_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_candidate_prime(x):
        for p in filter_primes:
            if x % p == 0:
                return False
        return True

    # We will iterate until deadline
    # Each iteration:
    # 1. Generate random odd number in range
    # 2. Check if prime
    # 3. If prime, check smoothness run
    
    # To avoid bias, we can also try a few specific large primes?
    # No, random is better for coverage.
    
    # Optimization: If we find a run of length k, we stop checking that p.
    # We want to maximize k.
    
    # Let's run a loop
    # We estimate ~10000 primality tests per second?
    # Miller-Rabin is fast.
    
    # We will try to find a p with k > best_k.
    
    # To improve chances, we can try to find p where p^2+1 is composite.
    # If p^2+1 is prime, k=0.
    # Probability p^2+1 is prime is ~ 1/log(p^2) ~ 1/(2 log p).
    # For p=10^6, 1/27 ~ 3.7%.
    # So 96% of primes have k>=1.
    
    # The bottleneck is finding a run of length > current best.
    
    while time.time() < deadline:
        # Generate random candidate
        # Range: 10^5 to 10^9
        # Use random.randrange(100001, 1000000000, 2) to get odd numbers
        # But we want primes.
        
        # Better: Generate random odd, check primality.
        # To speed up, we can use a "wheel" or just random odd.
        
        # Let's pick a random odd number
        x = random.randrange(100001, 1000000000, 2)
        
        # Quick filter
        if not is_candidate_prime(x):
            continue
            
        # Miller-Rabin
        if not _is_prime_mr(x):
            continue
            
        # x is a prime. Check smoothness.
        p = x
        p_sq = p * p
        k = 0
        # We only care if k > best_k
        # So we can stop checking as soon as we find a failure
        # Or if we reach best_k + 1 and it's smooth, we update best_k and continue.
        
        for i in range(1, best_k + 2): # Check up to best_k + 1
            if time.time() > deadline:
                break
            val = p_sq + i
            if _is_p_smooth(val, p, small_primes):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
                
    return {'p': best_p, 'k': best_k}
