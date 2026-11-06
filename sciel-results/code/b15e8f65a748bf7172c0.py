import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # 1. Precompute primes up to 10^6 for trial division and initial sieve
    limit_sieve = 10**6
    is_prime_sieve = [True] * (limit_sieve + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = False
    for i in range(2, int(limit_sieve**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, limit_sieve + 1, i):
                is_prime_sieve[j] = False
    
    small_primes = [i for i, prime in enumerate(is_prime_sieve) if prime]
    
    # 2. Miller-Rabin primality test
    def is_prime(n):
        if n < 2: return False
        if n < limit_sieve:
            return is_prime_sieve[n]
        # Deterministic Miller-Rabin for n < 2^64
        # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard Rho
    def pollard_rho(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
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

    # 4. Check if all prime factors of n are <= limit
    def all_factors_leq(n, limit):
        if n == 1:
            return True
        if n > limit:
            # If n is prime, it's a factor > limit.
            # If n is composite, it might have factors > limit, but also <= limit.
            # We must factor it to be sure.
            # However, if n > limit, and n is prime, return False.
            # If n is composite, we need to check its factors.
            pass
        
        if is_prime(n):
            return n <= limit
            
        # Try small primes first
        for p in small_primes:
            if p > limit:
                break
            if n % p == 0:
                if p > limit:
                    return False
                return all_factors_leq(n // p, limit)
        
        # If no small factor found, use Pollard Rho
        # But first, check if n is prime again (just in case small primes missed it? No, is_prime covers it)
        # Actually, if n is not prime and no small factor, it has large factors.
        d = pollard_rho(n)
        if d == n:
            # Retry or fallback
            return is_prime(n) and n <= limit
        
        return all_factors_leq(d, limit) and all_factors_leq(n // d, limit)

    # 5. Search for best p
    best_p = 2
    best_k = 1 # p=2, p^2+1=5 (prime > 2). Wait.
    # p=2: p^2+1 = 5. 5 > 2. So k=0?
    # The task says k >= 1.
    # "every integer p^2 + 1, ..., p^2 + k must have all prime factors at most p"
    # If p=2, p^2+1=5. Factor 5 > 2. So k cannot be 1.
    # So p=2 is invalid for k>=1.
    
    # Let's check p=3.
    # p^2+1 = 10. Factors 2,5. 5 > 3. Invalid.
    # p=5.
    # p^2+1 = 26. Factors 2,13. 13 > 5. Invalid.
    # p=7.
    # p^2+1 = 50. Factors 2,5. 5 <= 7. OK.
    # p^2+2 = 51. Factors 3,17. 17 > 7. Invalid.
    # So for p=7, k=1.
    
    # We need to find a p where k is larger.
    
    # Generate candidate primes
    # We can't iterate all primes up to 10^9.
    # Let's try a range.
    # Start from small primes, then jump.
    
    # Use the sieve to get primes up to limit_sieve
    candidates = small_primes.copy()
    
    # Add some larger primes?
    # We can generate more primes if we have time, but small primes are likely to have better "smoothness" properties relative to p?
    # Actually, for large p, the interval [p^2, p^2+k] is small compared to p^2.
    # The probability that a random number near p^2 is p-smooth is roughly 1/log(p^2) = 1/(2 log p).
    # So expected run length is small.
    # However, we are looking for the MAX k.
    # It might be that a specific large p has a lucky run.
    
    # Strategy:
    # 1. Check all primes up to 10^6 (100k primes).
    # 2. If time remains, check random primes up to 10^9.
    
    # Optimization:
    # For a given p, check k=1, 2, ...
    # Stop if time is running out.
    
    def check_p(p):
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > time_limit:
                return k
            n = base + k + 1
            if all_factors_leq(n, p):
                k += 1
            else:
                break
        return k

    # First pass: small primes
    for p in candidates:
        if time.time() - start_time > time_limit:
            break
        k = check_p(p)
        if k > best_k:
            best_k = k
            best_p = p
            
    # Second pass: random larger primes
    # Generate random odd numbers, check if prime, then check p
    # We need to ensure we don't duplicate work, but duplicates are okay if rare.
    
    while time.time() - start_time < time_limit:
        # Random number up to 10^9
        # Ensure it's not divisible by 2
        p_candidate = random.randint(3, 10**9)
        if p_candidate % 2 == 0:
            p_candidate += 1
            
        if is_prime(p_candidate):
            k = check_p(p_candidate)
            if k > best_k:
                best_k = k
                best_p = p_candidate
                
    return {'p': best_p, 'k': best_k}
