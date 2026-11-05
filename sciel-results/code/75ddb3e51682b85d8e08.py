import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Miller-Rabin
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic for n < 2^64
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
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
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
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
    return g

def smallest_prime_factor(n, limit=None):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    # Trial division up to 100
    for i in range(5, 100, 6):
        if n % i == 0:
            return i
        if n % (i + 2) == 0:
            return i + 2
    # Pollard's Rho
    while True:
        d = pollard_brent(n)
        if is_prime(d):
            return d
        # If d is composite, recurse? No, just return d and handle in smoothness check?
        # Actually, we want a prime factor.
        # Let's just return d and let the caller handle composite factors.
        return d

def is_p_smooth(n, p):
    if n <= p:
        return True
    if is_prime(n):
        return False  # n > p and prime
    # Find a factor
    d = smallest_prime_factor(n)
    if d > p:
        return False
    # d is a factor <= p
    # Check n//d
    return is_p_smooth(n // d, p)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Generate primes in a large range near 10^9
    # We'll use a segmented sieve
    limit = 10**9
    segment_size = 10**7
    
    # Sieve small primes for segmentation
    small_limit = int(math.isqrt(limit)) + 1
    is_prime_small = [True] * (small_limit + 1)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, math.isqrt(small_limit) + 1):
        if is_prime_small[i]:
            for j in range(i*i, small_limit + 1, i):
                is_prime_small[j] = False
    small_primes = [i for i, prime in enumerate(is_prime_small) if prime]
    
    # Process segments in descending order
    # Start from the largest segment
    # We can't go beyond 10^9
    start_seg = limit - segment_size + 1
    
    while start_seg > 2:
        if time.time() > deadline - 10:
            break
            
        # Segmented sieve for [start_seg, start_seg + segment_size - 1]
        # But we only want primes <= 10^9
        end_seg = min(start_seg + segment_size - 1, limit)
        
        # Initialize segment
        seg = [True] * (end_seg - start_seg + 1)
        
        for p in small_primes:
            if p * p > end_seg:
                break
            # Find first multiple of p in [start_seg, end_seg]
            start = max(p * p, ((start_seg + p - 1) // p) * p)
            for j in range(start, end_seg + 1, p):
                seg[j - start_seg] = False
                
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
                
            # Check k for this p
            k = 0
            while k < 200:
                if time.time() > deadline - 5:
                    break
                n = p * p + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
                    
            if k > best_k:
                best_k = k
                best_p = p
                
        start_seg -= segment_size
        
    return {'p': best_p, 'k': best_k}
