"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def now():
        return time.time()
    
    # Precomputed small primes for trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in small_primes:
            if n % p == 0:
                return n == p
        # Miller-Rabin for larger numbers
        # Deterministic for n < 3,317,044,064,679,887,385,961,981
        d = n - 1
        r = 0
        while d % 2 == 0:
            r += 1
            d //= 2
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
        if n % 5 == 0:
            return 5
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
    
    def factorize(n):
        if n == 1:
            return []
        if is_prime(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)
    
    def phi(n):
        if n == 1:
            return 1
        # Quick check for small primes
        if is_prime(n):
            return n - 1
        factors = factorize(n)
        unique_factors = set(factors)
        result = n
        for p in unique_factors:
            result -= result // p
        return result
    
    # Phase 1: Fast sieve-like computation for small range
    # We'll compute phi for 1 to N using a modified sieve
    N = 500000
    phi_vals = list(range(N + 1))
    for i in range(2, N + 1):
        if phi_vals[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi_vals[j] -= phi_vals[j] // i
    
    best_k = 1
    best_s = 1
    
    # Search for best run in small range
    for s in range(1, N - 200):
        if now() > DEADLINE:
            break
        seen = set()
        k = 0
        while k < 200 and s + k <= N:
            val = phi_vals[s + k]
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Random search in larger range
    # Use phi with Pollard's Rho for larger numbers
    random.seed(42)
    
    def check_run_large(s, max_k=200):
        seen = set()
        k = 0
        while k < max_k:
            if now() > DEADLINE:
                return k
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    # Try structured starts: multiples of small primes, etc.
    # Also random starts
    
    # First, try some structured starts near powers of 2, 3, etc.
    structured_starts = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        for power in range(1, 40):
            val = p ** power
            if val <= 10**12 - 200:
                structured_starts.append(val - 10)
                structured_starts.append(val)
                structured_starts.append(val + 10)
    
    structured_starts = sorted(set(s for s in structured_starts if s >= 1))
    
    for s in structured_starts:
        if now() > DEADLINE:
            break
        k = check_run_large(s)
        if k > best_k:
            best_k = k
            best_s = s
        # Check neighbors if good
        if k >= best_k - 2:
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                if now() > DEADLINE:
                    break
                s2 = s + delta
                if s2 >= 1 and s2 + 199 <= 10**12:
                    k2 = check_run_large(s2)
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
    
    # Random search
    while now() < DEADLINE:
        # Mix of uniform and biased sampling
        if random.random() < 0.5:
            s = random.randint(10**6, 10**12 - 200)
        else:
            # Biased towards numbers with many small prime factors
            # Sample a random number and multiply by small primes
            base = random.randint(10**3, 10**9)
            for _ in range(random.randint(1, 5)):
                base *= random.choice([2, 3, 5, 7, 11])
            if base > 10**12 - 200:
                base = base // random.choice([2, 3, 5, 7, 11])
            s = base + random.randint(-100, 100)
            s = max(1, min(s, 10**12 - 200))
        
        k = check_run_large(s)
        
        if k > best_k:
            best_k = k
            best_s = s
        
        # If we found a good run, search neighbors more thoroughly
        if k >= best_k - 3:
            for delta in range(-20, 21):
                if now() > DEADLINE:
                    break
                s2 = s + delta
                if s2 >= 1 and s2 + 199 <= 10**12:
                    k2 = check_run_large(s2)
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
    
    # Final validation
    if best_s + best_k - 1 > 10**12 or best_k < 1 or best_k > 200 or best_s < 1:
        return {'start': 1, 'length': 1}
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
