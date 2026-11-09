import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def phi(x):
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    best_k = 1
    best_s = 1
    
    # 2. Search in high range
    # We want to find a run of length 200.
    # Random sampling is good.
    # Range: [10^11, 10^12 - 200]
    low = 10**11
    high = 10**12 - 200
    
    # Try a few deterministic points first? Or just random.
    # Random is better to avoid bias.
    random.seed(42)
    
    # We can also try to verify if k=200 is possible.
    # If we find one, we return immediately.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        s = random.randint(low, high)
        
        # Check run of length 200 starting at s
        seen = set()
        k = 0
        valid = True
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                valid = False
                # Update best if k > best_k
                if k > best_k:
                    best_k = k
                    best_s = s
                break
            seen.add(val)
            k += 1
        
        if valid:
            # Found a run of length 200
            return {'start': s, 'length': 200}
        
        # If we broke early, we already updated best_k inside the loop if k > best_k
        # But what if k was smaller than best_k? No update needed.
        # What if k was larger but we didn't enter the 'if k > best_k' because we broke?
        # The 'if k > best_k' is inside the break block, so it runs.
        
        # Optimization: If we found a collision at k=1, it's very bad.
        # But for large s, collisions are rare.
        
        # If we are running out of time, we might want to do a more thorough search
        # for the best k found so far, but random sampling of 200-length blocks
        # is likely to find a full 200 block quickly if they exist.
        
        # If we haven't found 200, we might find a partial one.
        # The loop above updates best_k if a collision happens after k steps.
        
        # Let's add a check: if we haven't found 200, maybe we should try
        # to find the max k for a specific s more carefully?
        # No, the loop already finds the max k for that s (until collision).
        
        # We can also try scanning sequentially from a random start?
        # No, random starts are better for finding long runs in a sparse space.
        
        # If we get close to deadline, return best found.
        if attempts % 100 == 0:
             # Do nothing, just check time
             pass

    return {'start': best_s, 'length': best_k}
