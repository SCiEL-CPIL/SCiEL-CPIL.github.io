import time
import math
import random

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
_MR_WITNESSES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_WITNESSES:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    for attempt in range(20):
        y = random.randint(1, n - 1)
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
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g != n:
            return g
        # g == n, fallback
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                if g != n:
                    return g
                break
    # Last resort: trial division
    for i in range(5, 10000, 6):
        if n % i == 0:
            return i
        if n % (i + 2) == 0:
            return i + 2
    return n  # should not happen for composite n

def _spf_small(n):
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return p
    return None

def all_factors_leq(n, p):
    """Check if all prime factors of n are <= p. Returns True/False."""
    if n <= 1:
        return True
    if n <= p:
        return True
    if is_prime(n):
        return False
    d = _spf_small(n)
    if d is not None:
        return all_factors_leq(n // d, p)
    # Use Pollard-Brent
    d = pollard_brent(n)
    if d == n:
        # n is prime (should have been caught above)
        return False
    if d > p:
        return False
    return all_factors_leq(n // d, p) and all_factors_leq(n // d, p)
    
    # Actually we need to check both factors
    # Let me rewrite this properly

def all_factors_leq(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    if n <= p:
        return True
    if is_prime(n):
        return False
    d = _spf_small(n)
    if d is not None:
        return all_factors_leq(n // d, p)
    d = pollard_brent(n)
    if d == n:
        return False
    return all_factors_leq(d, p) and all_factors_leq(n // d, p)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s kill, 35s buffer)
    
    best_k = 0
    best_p = 2
    
    # First: try a set of promising candidate primes
    # Known good primes from literature/previous runs
    # Also try primes in ranges where smooth runs are more likely
    candidate_primes = []
    
    # Try primes around specific values
    # Based on the insight that p^2 + i being p-smooth is more likely
    # when p is in certain ranges
    
    # Generate a list of primes to try, prioritized
    # 1. Small primes (fast to check, low k)
    # 2. Medium primes
    # 3. Large primes near 10^9
    
    # Sieve for primes up to a reasonable limit for quick candidates
    # Then do segmented sieve for larger ranges
    
    # Quick sieve up to 10^6
    limit_small = 10**6
    is_prime_small = [True] * (limit_small + 1)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, math.isqrt(limit_small) + 1):
        if is_prime_small[i]:
            for j in range(i*i, limit_small + 1, i):
                is_prime_small[j] = False
    small_primes_list = [i for i, prime in enumerate(is_prime_small) if prime]
    
    # Check small primes first (fast, may find decent k)
    for p in small_primes_list:
        if time.time() > deadline - 20:
            break
        k = 0
        while k < 200:
            if time.time() > deadline - 20:
                break
            n = p * p + k + 1
            if all_factors_leq(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Now do segmented sieve for larger primes
    # Use a smaller segment size for better cache performance
    limit = 10**9
    segment_size = 5 * 10**6
    
    # Sieve small primes for segmentation (up to sqrt(10^9) ~ 31623)
    small_limit = int(math.isqrt(limit)) + 1
    is_prime_seg = [True] * (small_limit + 1)
    is_prime_seg[0] = is_prime_seg[1] = False
    for i in range(2, math.isqrt(small_limit) + 1):
        if is_prime_seg[i]:
            for j in range(i*i, small_limit + 1, i):
                is_prime_seg[j] = False
    seg_primes = [i for i, prime in enumerate(is_prime_seg) if prime]
    
    # Process segments from high to low
    start_seg = limit - segment_size + 1
    
    while start_seg > limit_small:
        if time.time() > deadline - 10:
            break
            
        end_seg = min(start_seg + segment_size - 1, limit)
        
        # Initialize segment
        seg = bytearray([1]) * (end_seg - start_seg + 1)
        
        for p in seg_primes:
            if p * p > end_seg:
                break
            start = max(p * p, ((start_seg + p - 1) // p) * p)
            for j in range(start, end_seg + 1, p):
                seg[j - start_seg] = 0
                
        # Collect primes in descending order
        primes_in_seg = []
        for i in range(len(seg)):
            if seg[i]:
                val = start_seg + i
                if val > 1:
                    primes_in_seg.append(val)
                    
        primes_in_seg.sort(reverse=True)
        
        for p in primes_in_seg:
            if time.time() > deadline - 5:
                break
                
            k = 0
            while k < 200:
                if time.time() > deadline - 5:
                    break
                n = p * p + k + 1
                if all_factors_leq(n, p):
                    k += 1
                else:
                    break
                    
            if k > best_k:
                best_k = k
                best_p = p
                
        start_seg -= segment_size
        
    return {'p': best_p, 'k': best_k}
