import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Precompute primes up to 10^6 for factorization
    limit = 10**6
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
        
    def is_p_smooth(n, p):
        # Check if all prime factors of n are <= p
        if n == 1:
            return True
        if n > p and is_prime(n):
            return False
        
        # Divide by small primes
        temp = n
        # Use primes up to sqrt(p) or until temp is 1
        # If p is small, just check all primes <= p
        if p < 10000:
            for q in primes:
                if q > p:
                    break
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            if temp > 1:
                if temp > p:
                    return False
                # temp is prime here (since we divided up to sqrt(temp) or q*q > temp)
                # Actually if q*q > temp, temp is prime.
                # If loop ended because q > p, temp might be composite but > p?
                # No, if q > p, we stopped. But we only need factors <= p.
                # If temp > 1 and temp > p, it has a factor > p?
                # If temp is prime > p, yes.
                # If temp is composite, its smallest factor is > p (since we divided by all <= p).
                # So temp > p implies not smooth.
                return temp <= p
            return True
            
        else:
            # p is large
            # Divide by primes up to sqrt(p)
            sqrt_p = int(math.isqrt(p))
            for q in primes:
                if q > sqrt_p:
                    break
                while temp % q == 0:
                    temp //= q
            
            if temp == 1:
                return True
                
            # Check if temp is prime
            if is_prime(temp):
                return temp <= p
                
            # temp is composite
            # It has no factors <= sqrt(p)
            # We need to check if it has a factor > p
            # This is equivalent to checking if temp has a factor q such that q > p
            # Since temp is composite, let q be smallest prime factor.
            # q > sqrt(p).
            # If temp = q * r, and q <= r, then q <= sqrt(temp).
            # We need to check if r > p.
            # r = temp / q.
            # So we need to find a prime factor q of temp (with q > sqrt(p)) such that temp/q > p.
            # This is equivalent to temp/q > p => q < temp/p.
            # So we look for prime factors q in (sqrt(p), temp/p).
            # If such q exists, temp is not smooth.
            # If no such q exists, then for all prime factors q, temp/q <= p.
            # This implies all factors are <= p?
            # If q is a prime factor, and temp/q <= p, then the cofactor is <= p.
            # But q itself might be > p?
            # If q > p, then temp/q < temp/p.
            # If temp/q <= p, then q >= temp/p.
            # So if q > p, then temp/p < q < temp/p? Contradiction unless temp/p >= p => temp >= p^2.
            # temp <= n = p^2+i. So temp can be >= p^2.
            
            # Simpler: Just factorize temp completely using Pollard Rho or trial division up to sqrt(temp)?
            # sqrt(temp) <= sqrt(n) ~ p.
            # So we need to check primes up to p.
            # We already checked up to sqrt(p).
            # We need to check primes in (sqrt(p), p].
            # If we find a factor q in this range, we divide it out.
            # If we divide it out and the remainder is 1 or a prime <= p, it's smooth.
            
            # Let's just trial divide temp by primes from sqrt(p) to p.
            # This might be slow if p is 10^9.
            # But temp is small? No, temp can be ~ p^2.
            
            # Alternative: Use the property that if temp is not smooth, it has a factor > p.
            # Let q be the smallest prime factor of temp.
            # If q > p, then temp is not smooth.
            # If q <= p, we divide temp by q.
            # Repeat.
            
            # So:
            # 1. Find smallest prime factor q of temp.
            # 2. If q > p, return False.
            # 3. Divide temp by q.
            # 4. Repeat.
            
            # How to find smallest prime factor of temp efficiently?
            # We know temp has no factors <= sqrt(p).
            # So smallest factor q > sqrt(p).
            # We can trial divide by primes starting from sqrt(p).
            
            current_prime_idx = 0
            for q in primes:
                if q <= sqrt_p:
                    continue
                if q * q > temp:
                    break
                if temp % q == 0:
                    # q is a factor
                    if q > p:
                        return False
                    while temp % q == 0:
                        temp //= q
                    if temp == 1:
                        return True
                    # Check if remaining temp is smooth
                    # Recursively? Or just check if temp is prime and <= p, or has small factors?
                    # Since we are in the loop, we can just continue?
                    # No, temp changed.
                    # Let's restart the check for temp?
                    # Or just return is_p_smooth(temp, p)?
                    # But this might be recursive.
                    # Let's just use a helper loop.
                    pass
                    
            # If loop finished without finding factors in (sqrt(p), sqrt(temp)]
            # Then temp is prime or 1.
            if temp > 1:
                return temp <= p
            return True

    best_k = 0
    best_p = 3
    
    # Search primes p
    # We can iterate through primes list up to some limit
    # Or generate primes on the fly
    
    # Let's iterate p from 3 to 10^5 first, then maybe larger
    # But we need p to be prime.
    
    for p in primes:
        if p > 10**9:
            break
        if time.time() > deadline:
            break
            
        # Quick check: if p^2+1 is not smooth, k=0
        # Calculate k
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        # Optional: if best_k is very high, we might stop?
        # But we want to maximize.
        
    return {'p': best_p, 'k': best_k}
