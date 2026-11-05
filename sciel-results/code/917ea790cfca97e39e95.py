import random
import time
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

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
            d = gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n):
    if n == 1:
        return []
    factors = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if n % p == 0:
            while n % p == 0:
                factors.append(p)
                n //= p
    if n > 1:
        # If n is prime, pollard_rho will fail or return n?
        # Standard pollard_rho returns a factor. If n is prime, it might loop or return n.
        # We should check if n is prime first? Miller-Rabin is fast.
        # Or just try rho. If rho returns n, it's likely prime or a power of a prime?
        # Actually, if n is prime, gcd(abs(x-y), n) will be 1 or n.
        # If it returns n, it failed.
        # Let's add a simple primality test for speed.
        def is_prime(n):
            if n < 2: return False
            if n in {2, 3}: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
            
        if is_prime(n):
            factors.append(n)
        else:
            d = pollard_rho(n)
            factors.extend(factor(d))
            factors.extend(factor(n // d))
    return factors

def phi(n):
    if n == 1:
        return 1
    factors = factor(n)
    unique_factors = set(factors)
    result = n
    for p in unique_factors:
        result -= result // p
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    best_k = 1
    best_s = 1
    
    # Memoization for phi
    phi_cache = {}
    
    def get_phi(n):
        if n in phi_cache:
            return phi_cache[n]
        val = phi(n)
        phi_cache[n] = val
        return val

    # 1. Sieve for small numbers (1 to 200,000)
    MAX_SMALL = 200_000
    phi_small = list(range(MAX_SMALL + 1))
    for i in range(2, MAX_SMALL + 1):
        if phi_small[i] == i:
            for j in range(i, MAX_SMALL + 1, i):
                phi_small[j] -= phi_small[j] // i
                
    # Search in small range
    for s in range(1, MAX_SMALL - 200):
        if time.time() > deadline: break
        seen = set()
        k = 0
        while s + k <= MAX_SMALL and phi_small[s + k] not in seen:
            seen.add(phi_small[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}

    # 2. Random search
    # To avoid re-computing phi for numbers that are close to each other in different random picks,
    # the cache helps.
    
    while time.time() < deadline:
        s = random.randint(1, 10**12 - 200)
        seen = set()
        k = 0
        while k < 200:
            val = get_phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
