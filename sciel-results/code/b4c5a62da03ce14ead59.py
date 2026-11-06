import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    def now():
        return time.time() - start_time

    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    def phi(x):
        if x <= 1:
            return 1 if x == 1 else 0
        result = x
        y = x
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    max_n = 10**12
    best_k = 1
    best_s = 1
    
    # Phase 1: Exhaustive search in [1, 500000]
    N1 = 500000
    values = [0] * (N1 + 1)
    for x in range(1, N1 + 1):
        values[x] = phi(x)
        
    for s in range(1, N1 - 199):
        seen = set()
        k = 0
        while s + k <= N1 and values[s + k] not in seen:
            seen.add(values[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s

    if now() > DEADLINE - 10:
        return {'start': best_s, 'length': best_k}

    # Phase 2: Random search
    random.seed(42)
    
    while now() < DEADLINE - 5:
        # Sample s
        # Bias towards smaller numbers? 
        # phi(n) ~ n * 6/pi^2. 
        # Collisions are more likely when values are close.
        # For n ~ 10^12, phi(n) ~ 6*10^11.
        # Window width 200.
        # Values change by ~ 6*10^9 per step.
        # Range of phi values in window is ~ 10^12.
        # Number of possible phi values is huge.
        # Random search should work.
        
        s = random.randint(100000, max_n - 200)
        
        seen = set()
        k = 0
        for i in range(200):
            n_val = s + i
            if n_val > max_n:
                break
            v = phi(n_val)
            if v in seen:
                break
            seen.add(v)
            k += 1
            
        if k > best_k:
            best_k = k
            best_s = s
            
            # Local refinement
            if k > 5:
                for delta in range(-10, 11):
                    if now() > DEADLINE - 5:
                        break
                    ns = s + delta
                    if ns < 1 or ns + 199 > max_n:
                        continue
                    seen2 = set()
                    k2 = 0
                    for i in range(200):
                        n_val = ns + i
                        if n_val > max_n:
                            break
                        v = phi(n_val)
                        if v in seen2:
                            break
                        seen2.add(v)
                        k2 += 1
                    if k2 > best_k:
                        best_k = k2
                        best_s = ns

    return {'start': best_s, 'length': best_k}
