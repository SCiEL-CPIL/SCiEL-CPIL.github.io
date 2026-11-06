import math
import time
import random

# Precompute small primes for trial division
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return n in (2, 3)
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
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
    """Brent's variant of Pollard's rho - faster than basic rho."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    y, c, m = 2, 1, 128
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
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
    return g

def largest_prime_factor(n):
    if n == 1:
        return 1
    if n < 0:
        n = -n
    # Trial division by small primes
    max_f = 1
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            max_f = p
            while n % p == 0:
                n //= p
    if n == 1:
        return max_f
    if is_prime(n):
        return max(n, max_f)
    # Recursively find factors
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > max_f:
                max_f = m
        else:
            d = pollard_brent(m)
            if d == m:
                # Shouldn't happen, but handle it
                if m > max_f:
                    max_f = m
            else:
                stack.append(d)
                stack.append(m // d)
    return max_f

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    best_k = 0
    best_p = 2

    # Sieve primes up to 10^7
    limit = 10**7
    is_prime_sieve = bytearray(b'\x01') * (limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime_sieve[i]]

    # Check each prime p
    for p in primes:
        if time.time() > deadline:
            break
        
        p2 = p * p
        
        # Quick rejection: if p^2+1 is prime and > p, k=0
        n1 = p2 + 1
        if is_prime(n1):
            continue
            
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p2 + i
            
            # Quick check: is n_val prime? If so and n_val > p, fail
            if is_prime(n_val):
                if n_val > p:
                    break
                else:
                    k += 1
                    continue
            
            # Find largest prime factor
            lf = largest_prime_factor(n_val)
            
            if lf > p:
                break
            
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p

    # Try some larger primes beyond the sieve
    if time.time() < deadline - 10:
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            p_candidate = random.randint(10**7, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
            if is_prime(p_candidate):
                p = p_candidate
                p2 = p * p
                n1 = p2 + 1
                if is_prime(n1):
                    continue
                k = 0
                for i in range(1, 201):
                    if time.time() > deadline:
                        break
                    n_val = p2 + i
                    if is_prime(n_val):
                        if n_val > p:
                            break
                        else:
                            k += 1
                            continue
                    lf = largest_prime_factor(n_val)
                    if lf > p:
                        break
                    k += 1
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
