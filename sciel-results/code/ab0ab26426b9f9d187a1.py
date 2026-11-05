import time
import math
import random

# Sieve of Eratosthenes to get primes up to 10^6
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

_SMALL_PRIMES = sieve(1000000)
_SMALL_PRIME_SET = set(_SMALL_PRIMES)

# Deterministic Miller-Rabin for 64-bit integers
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

def is_p_smooth(n, p):
    """
    Check if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    
    # Trial division by small primes up to min(p, 10^5)
    # This quickly reduces n
    limit = min(p, 100000)
    for q in _SMALL_PRIMES:
        if q > limit:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Otherwise, n is composite.
    # We need to check if all its prime factors are <= p.
    # This is equivalent to: does n have any prime factor > p?
    # If n has a prime factor > p, then n is not p-smooth.
    # We can find a factor d. If d > p and is_prime(d) is False, we must check further.
    # Actually, simpler: recursively factorize and check max factor.
    # But we can short-circuit: if we find a factor d such that d > p and d is prime, return False.
    # If d is composite, we need to check its factors.
    
    # Let's use a stack-based factorization check
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            if x > p:
                return False
            continue
        
        # Try to find a factor
        d = None
        # Try small primes again? No, we already did that for the initial n.
        # But x might be smaller than n, so it might have small factors we haven't checked for x specifically?
        # Actually, if we divided out all small primes from n, x (a factor of n) also has no small prime factors.
        # So we can go straight to Pollard-Brent.
        
        # Optimization: If x is a perfect square, check sqrt
        r = int(math.isqrt(x))
        if r * r == x:
            stack.append(r)
            continue
            
        d = pollard_brent(x)
        stack.append(d)
        stack.append(x // d)
        
    return True

def construct():
    deadline = time.time() + 110
    best_k = 0
    best_p = 0
    
    # We iterate through primes p.
    # For each p, we check k=1, 2, ...
    # We stop checking k for a specific p when we find a non-smooth number or k=200.
    
    # To be efficient, we should check smaller p first, as they are "cheaper" to verify?
    # Actually, larger p means larger n, which is slower to factor.
    # But larger p means the smoothness condition is "looser" (more numbers are p-smooth).
    # However, the probability of a *run* of length k is roughly constant or decreasing?
    # Let's just iterate p in increasing order.
    
    # We need a list of primes to iterate.
    # We have primes up to 10^6.
    # For p > 10^6, we generate them on the fly.
    
    for p in _SMALL_PRIMES:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
            
        k = 0
        base = p * p
        while k < 200:
            if time.time() > deadline:
                break
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we have time, try some larger primes
    # Generate primes > 10^6
    last_p = _SMALL_PRIMES[-1]
    while True:
        if time.time() > deadline:
            break
        last_p += 2
        if last_p > 10**9:
            break
        if not is_prime(last_p):
            continue
            
        p = last_p
        k = 0
        base = p * p
        while k < 200:
            if time.time() > deadline:
                break
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    if best_k < 1:
        # Fallback: find any valid p,k
        # p=7, k=1 is valid.
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
