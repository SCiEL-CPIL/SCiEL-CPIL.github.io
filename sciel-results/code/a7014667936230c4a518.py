import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    # 1. Generate primes up to 10^6 using NumPy for speed
    limit_p = 10**6
    # NumPy sieve
    is_prime = np.ones(limit_p + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, v in enumerate(is_prime) if v])
    
    # 2. Efficient phi calculation using vectorized modulo
    def phi(x):
        if x <= 1:
            return 1
        result = x
        n = x
        
        # Vectorized check for divisibility by all primes
        # This is fast because it's done in C
        # However, we only need to check primes up to sqrt(n)
        # But n changes.
        # Strategy: 
        # 1. Check small primes explicitly (fast path)
        # 2. Then use vectorized check for remaining primes?
        # Actually, n % primes for 78k primes is ~0.1ms?
        # Let's stick to the loop but optimize it?
        # Or just use the vectorized approach for the initial check.
        
        # A hybrid approach:
        # Check small primes (2, 3, 5, 7, 11, 13) manually.
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                result -= result // p
                while n % p == 0:
                    n //= p
        
        # If n is still large, use vectorized check on the rest of the primes
        # We need primes > 37
        mask = primes > 37
        large_primes = primes[mask]
        
        # We only need to check up to sqrt(n)
        # But n is changing.
        # Simple vectorized check:
        # rems = n % large_primes
        # This allocates an array of size ~78k.
        # It's fast in C.
        
        if n > 1:
            # Optimization: if n is small enough, sqrt(n) is small.
            # But n starts up to 10^12.
            # Let's just do the vectorized check.
            # It might be overkill if n becomes small quickly.
            # But for random n, it stays large.
            
            # Check if n is prime?
            # If n is prime, result -= result // n.
            # If n is composite, we need to factor it.
            
            # Vectorized modulo
            # Note: n is Python int. large_primes is np array.
            # n % large_primes works.
            try:
                rems = n % large_primes
                div_indices = np.where(rems == 0)[0]
                
                if len(div_indices) == 0:
                    # n is prime (or 1)
                    if n > 1:
                        result -= result // n
                else:
                    # Process the divisors
                    for idx in div_indices:
                        p = large_primes[idx]
                        result -= result // p
                        while n % p == 0:
                            n //= p
                        # If n becomes 1, break
                        if n == 1:
                            break
                        # If n is now smaller than p*p, it's prime?
                        # Not necessarily, but if no more divisors found in the vectorized check
                        # (which checked ALL large primes), then n is prime.
                        # Wait, the vectorized check checked ALL large primes.
                        # So if we found a divisor, we removed it.
                        # Are there other divisors?
                        # Yes, n might have multiple distinct prime factors.
                        # The loop `for idx in div_indices` handles distinct primes.
                        # But `n` changes.
                        # Does `rems` need to be updated?
                        # No, `rems` was calculated for the *original* n.
                        # If n was divisible by p, we removed p.
                        # Is n still divisible by another prime q in large_primes?
                        # Yes, if the original n was divisible by q.
                        # And `rems` would have shown q as a divisor.
                        # So `div_indices` contains ALL distinct prime factors of the *original* n (that are in large_primes).
                        # So we can just iterate through them.
                        
                    if n > 1:
                        # If n > 1 after removing all found prime factors,
                        # and we checked ALL primes up to 10^6,
                        # then n must be prime (since n <= 10^12, if it had two factors > 10^6, n > 10^12).
                        # So n is prime.
                        result -= result // n
            except:
                # Fallback to standard loop if vectorized fails or is too slow/memory error
                for p in large_primes:
                    if p * p > n:
                        break
                    if n % p == 0:
                        result -= result // p
                        while n % p == 0:
                            n //= p
                if n > 1:
                    result -= result // n
                    
        return result

    # 3. Search
    best_start = 1
    best_len = 1
    
    def check_run(s, target_len=200):
        seen = set()
        k = 0
        ph = phi
        for i in range(target_len):
            val = ph(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    random.seed(42)
    attempts = 0
    max_samples = 1000
    
    while time.time() - start_time < DEADLINE and attempts < max_samples:
        s = random.randint(1, 10**12 - 200)
        k = check_run(s, 200)
        
        if k > best_len:
            best_len = k
            best_start = s
            if best_len == 200:
                break
        
        attempts += 1

    return {'start': best_start, 'length': best_len}
