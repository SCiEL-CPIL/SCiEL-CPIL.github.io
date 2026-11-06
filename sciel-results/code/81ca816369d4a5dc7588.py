import math
import random

# Generate small primes for trial division
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

SMALL_PRIMES = sieve(10000)

def is_prime(n):
    if n < 2:
        return False
    if n in SMALL_PRIMES:
        return True
    for p in SMALL_PRIMES:
        if p * p > n:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin for larger numbers
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n, result=None):
    if result is None:
        result = []
    if n == 1:
        return result
    if is_prime(n):
        result.append(n)
        return result
    
    # Trial division by small primes
    for p in SMALL_PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                result.append(p)
                n //= p
            if n == 1:
                return result
            if is_prime(n):
                result.append(n)
                return result
    
    # Pollard's Rho for the remainder
    d = pollard_rho(n)
    factor(d, result)
    factor(n // d, result)
    return result

def phi(n):
    if n == 1:
        return 1
    factors = factor(n)
    unique_primes = set(factors)
    result = n
    for p in unique_primes:
        result -= result // p
    return result

def construct():
    # Deterministic search
    # Max k is likely small. Let's try up to 50.
    # Start from high numbers to maximize the chance of distinctness.
    for k in range(50, 0, -1):
        max_start = 10**12 - k + 1
        # Check a few deterministic spots
        # 1. Near the top
        # 2. Random-ish but deterministic offsets
        offsets = [0, 1, 2, 100, 1000, 12345, 99999, 1000000]
        candidates = set()
        for off in offsets:
            s = max_start - off
            if s >= 1:
                candidates.add(s)
        
        # Add some fixed high numbers
        candidates.add(10**12 - 1)
        candidates.add(10**11)
        
        for s in candidates:
            seen = set()
            valid = True
            for i in range(k):
                val = phi(s + i)
                if val in seen:
                    valid = False
                    break
                seen.add(val)
            
            if valid:
                return {'start': s, 'length': k}
                
    return {'start': 1, 'length': 1}
