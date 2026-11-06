import math
import time
import random

def construct():
    deadline = time.time() + 115.0

    # Sieve primes up to 10^6
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]

    limit = 10**6
    primes = sieve_primes(limit)
    
    # Map prime to index for binary search or just slice
    # We need primes <= p. Since p is in primes, we can find its index.
    # Or just iterate primes and break when prime > p.

    def is_p_smooth_trial(n, p, primes_list):
        """
        Check if n is p-smooth using trial division by primes in primes_list.
        Assumes primes_list contains all primes <= p.
        """
        temp_n = n
        for sp in primes_list:
            if sp > p:
                break
            if sp * sp > temp_n:
                # If sp^2 > temp_n, then temp_n is prime (or 1)
                # If temp_n > 1, it's a prime factor.
                # Check if it's <= p
                if temp_n > 1:
                    return temp_n <= p
                else:
                    return True
            if temp_n % sp == 0:
                # Divide out all factors of sp
                while temp_n % sp == 0:
                    temp_n //= sp
                if temp_n == 1:
                    return True
        # If we get here, we've checked all primes <= p
        # If temp_n > 1, it has no prime factors <= p
        # So it's not p-smooth (unless temp_n == 1)
        return temp_n == 1

    best_k = 0
    best_p = 2

    # Search small primes
    for idx, p in enumerate(primes):
        if time.time() > deadline - 10:
            break
        
        # Get primes <= p. Since primes is sorted, it's primes[:idx+1]
        # Slicing creates a new list, which is slow.
        # Better to pass the list and the limit index.
        
        base = p * p
        k = 0
        for i in range(1, 201):
            n = base + i
            # Check smoothness
            # Use is_p_smooth_trial but optimized to avoid slicing
            temp_n = n
            smooth = True
            for j in range(idx + 1):
                sp = primes[j]
                if sp * sp > temp_n:
                    if temp_n > 1:
                        if temp_n > p:
                            smooth = False
                    break
                if temp_n % sp == 0:
                    while temp_n % sp == 0:
                        temp_n //= sp
                    if temp_n == 1:
                        break
            else:
                # Loop finished without break
                if temp_n > 1:
                    smooth = False
            
            if smooth:
                k = i
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
