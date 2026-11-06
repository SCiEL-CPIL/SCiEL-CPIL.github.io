"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def is_prime_mr(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic bases for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i : limit+1 : i] = bytearray(len(is_p[i*i : limit+1 : i]))
    return [i for i, v in enumerate(is_p) if v]

def construct():
    # We search for p up to a reasonable limit.
    # The current best was found with p < 100,000 likely.
    # Let's extend the search a bit and optimize the smoothness check.
    
    # Precompute primes up to a limit.
    # If p is up to 10^6, we need primes up to 10^6.
    # Let's use 10^6 as a safe upper bound for p search initially, 
    # but we can increase if time permits. 
    # Given 120s limit, we can afford more.
    
    LIMIT_P = 10**6
    primes = sieve_primes(LIMIT_P)
    
    best_k = 0
    best_p = 2
    
    # We iterate through primes p
    for p in primes:
        if p > LIMIT_P:
            break
            
        # Optimization: if p is small, k is likely small.
        # We check i from 1 to 200.
        
        k = 0
        for i in range(1, 201):
            n = p * p + i
            
            # Check if n is p-smooth.
            # Strategy: Trial divide by primes q <= p.
            # We can use the precomputed primes list.
            # We only need primes up to p.
            # Since primes is sorted, we can find the index of p.
            # To avoid binary search every time, we can maintain a pointer or just use a slice.
            # However, slicing creates a new list.
            # We can just iterate and break when q > p.
            
            is_smooth = True
            temp_n = n
            
            # We only need to trial divide by primes q <= p.
            # Note: if temp_n becomes 1, we are done.
            # If temp_n > 1 after dividing by all q <= p, then temp_n is either 1, a prime > p, or a product of primes > p.
            # Since we divided by all primes <= p, any remaining factor must be > p.
            # So if temp_n > 1, it has a prime factor > p, so it's NOT p-smooth.
            # Wait, is that true?
            # If n = q1 * q2 where q1 <= p and q2 > p.
            # We divide by q1. temp_n becomes q2.
            # We stop when q > p.
            # Then temp_n = q2 > p. So is_smooth = False.
            # If n is prime and n <= p, we divide by nothing? No, we divide by primes <= p.
            # If n is prime and n <= p, then n is in the list of primes.
            # We will divide n by n, temp_n becomes 1. is_smooth = True.
            # If n is prime and n > p, we don't divide by it. temp_n = n > p. is_smooth = False.
            # So the logic holds: after trial division by all primes <= p, if temp_n == 1, it is p-smooth.
            
            for q in primes:
                if q > p:
                    break
                if temp_n % q == 0:
                    while temp_n % q == 0:
                        temp_n //= q
                    if temp_n == 1:
                        break
                # Optional optimization: if q*q > temp_n, then temp_n is prime.
                # But we must still check if that prime is <= p.
                # Actually, if q*q > temp_n, then temp_n is prime (or 1).
                # If temp_n is prime, is it <= p?
                # We are iterating q up to p.
                # If we reach q such that q*q > temp_n, and we haven't found a factor,
                # then temp_n is prime.
                # We need to check if temp_n <= p.
                # But we are inside the loop `for q in primes: if q > p: break`.
                # So we only check primes up to p.
                # If temp_n is prime and temp_n <= p, it should have been found in the division?
                # Yes, if temp_n is prime and <= p, it is in the list `primes` up to p.
                # So it would have been divided out.
                # Thus, if we finish the loop (q > p) and temp_n > 1, then temp_n has no prime factors <= p.
                # Therefore temp_n has a prime factor > p (itself, or a composite with factors > p).
                # So it is NOT p-smooth.
                
            if temp_n == 1:
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
