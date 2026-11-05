"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
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
                if g > 1:
                    break
        if g != n:
            return g

def factorize(n):
    factors = []
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            factors.append(x)
            continue
        d = pollard_brent(x)
        stack.append(d)
        stack.append(x // d)
    return factors

def largest_prime_factor(n):
    if n <= 1:
        return 1
    fs = factorize(n)
    return max(fs)

def check_smooth(p, start, count):
    """Check if all numbers start..start+count-1 have all prime factors <= p."""
    for i in range(count):
        n = start + i
        if n <= 1:
            continue
        lf = largest_prime_factor(n)
        if lf > p:
            return False
    return True

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    best_k = 1
    best_p = 2
    
    # Try some known good candidates and larger primes
    # First, quickly test small primes to get a baseline
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if time.time() > deadline:
            break
        k = 0
        while k < 200:
            n = p * p + k + 1
            lf = largest_prime_factor(n)
            if lf <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Try larger primes: search in ranges
    # For p > 200, p^2+i has cofactor < p+1 for any prime factor > p
    # Use random sampling of large primes
    random.seed(42)
    
    # Try primes in various ranges
    search_ranges = [
        (100, 500),
        (500, 2000),
        (2000, 10000),
        (10000, 100000),
        (100000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() > deadline:
            break
        # Sample primes in this range
        num_samples = 50
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            # Find next prime
            while not is_prime(p):
                p += 2
                if p > hi:
                    break
            if p > hi or p > 10**9:
                continue
            # Check smoothness
            k = 0
            while k < 200:
                if time.time() > deadline:
                    break
                n = p * p + k + 1
                lf = largest_prime_factor(n)
                if lf <= p:
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                # If we found a good one, try nearby primes
                for dp in [2, 4, 6, 8, 10, -2, -4, -6, -8, -10]:
                    if time.time() > deadline:
                        break
                    p2 = best_p + dp
                    if p2 < 2 or p2 > 10**9:
                        continue
                    if not is_prime(p2):
                        continue
                    k2 = 0
                    while k2 < 200:
                        if time.time() > deadline:
                            break
                        n = p2 * p2 + k2 + 1
                        lf = largest_prime_factor(n)
                        if lf <= p2:
                            k2 += 1
                        else:
                            break
                    if k2 > best_k:
                        best_k = k2
                        best_p = p2
    
    if best_k < 1:
        best_k = 1
        best_p = 2
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
