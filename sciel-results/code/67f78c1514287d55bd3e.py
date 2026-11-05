import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0
    
    # Sieve of Eratosthenes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def phi(n):
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    # Try to find a run of length 200
    max_k = 200
    max_s = 10**12 - max_k + 1
    
    # Try a few random starts, biased towards large numbers
    # Large numbers have more "complex" phi values, less likely to collide with small ones?
    # Actually, just random is fine.
    
    best_result = {'start': 1, 'length': 1}
    
    # Deterministic check near the end first, as it's a good candidate
    # Then random
    candidates = [10**12 - 199, 10**12 - 10000, 10**11]
    random.seed(42)
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        s = random.randint(1, max_s)
        candidates.append(s)
        
    for s in candidates:
        if time.time() > DEADLINE:
            break
        if s < 1 or s > max_s:
            continue
            
        vals = []
        seen = set()
        valid = True
        for i in range(max_k):
            v = phi(s + i)
            if v in seen:
                valid = False
                break
            seen.add(v)
            vals.append(v)
            
        if valid:
            return {'start': s, 'length': max_k}
            
    # Fallback: find max k in a smaller range if 200 fails (unlikely)
    # Or just return a small valid run
    # Let's try to find the best run in a small range to ensure validity if something goes wrong
    # But 200 should work.
    
    # If we reach here, we didn't find 200.
    # Try to find a smaller run?
    # For safety, return a known small run.
    # We can quickly scan 1..1000
    # But we already spent time.
    # Just return 1,1 which is always valid.
    # Or better, try to find a run of 100?
    
    # Let's do a quick scan for a run of 100 if 200 failed.
    # Actually, if 200 failed, it's due to time or bad luck.
    # Given the deadline, just return the best found.
    
    # If no candidate worked, return default
    return best_result
