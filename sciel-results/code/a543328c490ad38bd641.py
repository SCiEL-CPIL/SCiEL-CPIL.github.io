import math
import time

def construct():
    # 1. Sieve of Eratosthenes
    def get_primes(n):
        sieve = bytearray([True]) * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    # Generate primes up to 10^6
    MAX_P = 10**6
    primes = get_primes(MAX_P)
    
    # Precompute index of p in primes for quick lookup? 
    # Or just use bisect.
    import bisect
    
    best_k = 0
    best_p = 2
    start_time = time.time()
    timeout = 115

    # Map prime to its index in the list for O(1) access to the sublist of primes <= p
    # Actually, bisect_right(primes, p) gives the count of primes <= p.
    
    for p in primes:
        if time.time() - start_time > timeout:
            break
            
        # We only care if we can beat best_k
        # If best_k is already high, we might skip? No, we want to maximize.
        
        k = 0
        # Index of last prime <= p
        idx = bisect.bisect_right(primes, p)
        # primes[0:idx] are the primes <= p
        
        for i in range(1, 201):
            N = p * p + i
            # Check if N is p-smooth
            # Trial divide by primes[0:idx]
            temp = N
            for q in primes[:idx]:
                if q * q > temp:
                    break
                if temp % q == 0:
                    while temp % q == 0:
                        temp //= q
            if temp != 1:
                # temp is a prime factor > p (or 1 if N was 1, but N > 1)
                # Actually if temp > 1, it has a prime factor > p?
                # Yes, because we divided out all primes <= p.
                # So temp must be prime and > p, or composite with factors > p.
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
