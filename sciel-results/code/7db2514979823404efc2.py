import math
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Bases for deterministic MR for 64-bit integers
        # Actually, for n < 2^64, bases [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] are sufficient.
        # But we only deal with n up to ~10^12 (p^2+k), so fewer bases might suffice.
        # For n < 3,474,749,660,383, bases [2, 3, 5, 7, 11, 13, 17] are enough.
        # Our n is at most (10^6)^2 + 200 = 10^12, which is well within range.
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Sieve of Eratosthenes
    def sieve(limit):
        is_prime_arr = [True] * (limit + 1)
        is_prime_arr[0] = is_prime_arr[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime_arr[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime_arr[j] = False
        return [i for i, prime in enumerate(is_prime_arr) if prime]

    # Determine sieve limit based on time, but start with a reasonable guess
    # Let's try up to 2,000,000 first. If it's too slow, we'll stop early.
    sieve_limit = 2000000
    primes = sieve(sieve_limit)
    
    best_k = 0
    best_p = 2
    
    # Check small primes manually if not in list? 2 is in list.
    # Iterate through primes
    for p in primes:
        if time.time() - start_time > timeout:
            break
            
        # Check run length for this p
        k = 0
        # We want to maximize k.
        # p^2 + 1, ..., p^2 + k
        # If k is large, it takes longer.
        # We can set a cap on k for the search if time is tight, but we want to maximize.
        # Let's just check until failure.
        
        # Optimization: if p is small, k is likely small.
        # If p is large, k might be larger.
        
        # Pre-calculate p^2
        p2 = p * p
        
        # We need primes up to max(p, sqrt(p^2+k)).
        # Our primes list goes up to 2,000,000.
        # If p <= 2,000,000, we have all primes <= p.
        # sqrt(p^2+k) is approx p. So we have all needed primes.
        
        while True:
            if time.time() - start_time > timeout:
                break
                
            n = p2 + k + 1
            # Check if n is p-smooth
            r = n
            # Limit for trial division
            # We divide by primes <= min(p, sqrt(n))
            limit = min(p, int(n**0.5) + 1)
            
            smooth = True
            for q in primes:
                if q > limit:
                    break
                while r % q == 0:
                    r //= q
            
            if r == 1:
                # Smooth so far
                k += 1
                # Optional: cap k to avoid infinite loops if bug, but task says k<=200
                if k > 200:
                    break
                continue
            else:
                # r > 1
                # Check if r is prime
                if is_prime(r):
                    if r <= p:
                        k += 1
                        if k > 200:
                            break
                        continue
                    else:
                        smooth = False
                else:
                    smooth = False
                
                if not smooth:
                    break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
