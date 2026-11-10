import random
import time
import math

def construct():
    # Deadline guard
    deadline = time.time() + 110  # 10s margin from 120s limit
    
    # 1. Sieve primes up to 10^6
    # sqrt(10^12) = 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Fast phi computation using precomputed primes
    def phi(x):
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    # 3. Search for a window of length 200 with distinct phi values
    # We aim for k=200 immediately as it is the max allowed.
    # We search in a high range to minimize collisions.
    max_k = 200
    # Search range: [10^11, 10^12 - 200]
    low_start = 10**11
    high_start = 10**12 - max_k
    
    # We can also try to find a longer run if possible, but max is 200.
    # So we just need to find ANY run of 200.
    
    best_s = 1
    best_k = 1
    
    # Try a few fixed good starts first? No, random is better to avoid bad spots.
    # But random might be slow if we get unlucky. 
    # Let's try a few random starts.
    
    # To ensure we don't miss a better result if 200 is rare (it's not, but just in case),
    # we can check if we found 200 and return.
    
    attempts = 0
    while time.time() < deadline and attempts < 100:
        attempts += 1
        s = random.randint(low_start, high_start)
        
        # Check if phi(s) ... phi(s + max_k - 1) are distinct
        seen = set()
        valid = True
        current_k = 0
        for i in range(max_k):
            n = s + i
            p_val = phi(n)
            if p_val in seen:
                # Collision found, record the length up to here
                # But we want max length. If we hit collision at i, length is i.
                # However, we want to maximize k. If we can't get 200, we try another s.
                # It is highly likely that most s give 200.
                valid = False
                break
            seen.add(p_val)
            current_k = i + 1
        
        if valid:
            return {'start': s, 'length': max_k}
        
        # If we failed, we might have found a run of length current_k.
        # But usually we just restart with a new s.
        # We can track the best_k found so far if we don't find 200.
        if current_k > best_k:
            best_k = current_k
            best_s = s

    # If we run out of time or attempts, return best found
    # It is extremely likely we found 200 in the first few attempts.
    # If not, best_k might be less than 200.
    return {'start': best_s, 'length': best_k}
