"""Improved program: Efficient smoothness check and wider search."""
# EVOLVE-BLOCK-START

def construct():
    import math
    import time

    # ---------- Sieve of Eratosthenes ----------
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    # ---------- Smoothness Check ----------
    def is_p_smooth(x, p, primes):
        """
        Returns True if all prime factors of x are <= p.
        """
        if x == 1:
            return True
        if x <= p:
            return True
        
        temp = x
        # We only need to check primes up to min(p, sqrt(x))
        # But we can break early if temp becomes 1
        for q in primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
            if temp == 1:
                return True
        
        # If temp > 1, it must be prime (since we divided out all factors up to sqrt(temp) initially? 
        # No, we divided out factors up to sqrt(original x) or p.
        # If temp > 1 and temp <= p, it is smooth.
        # If temp > p, it is not smooth (it has a factor > p).
        return temp <= p

    # ---------- Search ----------
    best_k = 0
    best_p = 7 # Fallback
    start_time = time.time()
    TIME_LIMIT = 110.0
    
    # Generate primes up to a reasonable limit for trial division
    # We need primes up to sqrt(p^2+200) ~ p.
    # If we search p up to 10^6, we need primes up to 10^6.
    PRIME_LIMIT = 1_000_000
    primes = sieve_primes(PRIME_LIMIT)
    
    # Search for p in the range [2, PRIME_LIMIT]
    # We can skip very small p as they are unlikely to have long runs
    # But we must check them to be safe.
    
    # To speed up, we can iterate primes directly
    for p in primes:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        p2 = p * p
        k = 0
        # Check up to 200 offsets
        for offset in range(1, 201):
            val = p2 + offset
            if not is_p_smooth(val, p, primes):
                break
            k = offset
            
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we found a good k, return it.
    # If best_k is 0, we need to ensure we return a valid k >= 1.
    # The problem states 1 <= k.
    # If no p yields k >= 1, we are in trouble.
    # But p=7 yields k=1.
    # So best_k should be at least 1 if we check p=7.
    
    # If the loop didn't run or found nothing, fallback to known small prime
    if best_k == 0:
        best_p = 7
        best_k = 1
        
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
