import math
import random

def is_prime(n):
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Deterministic bases for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def construct():
    # We will search for primes p.
    # To be efficient, we generate primes up to a limit for the sieve.
    # We need primes up to sqrt(p_max).
    # Let's assume we search p up to 10^7. sqrt(10^7) ~ 3162.
    # Let's generate primes up to 10000 to be safe for p up to 10^8.
    
    limit_sieve = 10000
    sieve = [True] * (limit_sieve + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit_sieve ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit_sieve + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, limit_sieve + 1) if sieve[i]]
    
    best_k = 0
    best_p = 2
    
    # Search range for p
    # We can't iterate all integers. We need primes.
    # We can use a segmented sieve or just check primality.
    # Given time limit, let's search p up to 100,000 first with the fast check,
    # then maybe extend if time allows.
    
    # Actually, let's just iterate p from 2 to 200,000.
    # For each p, check if prime. If not, skip.
    # If prime, perform the sieve check.
    
    max_p_search = 200000
    
    for p in range(2, max_p_search + 1):
        if not is_prime(p):
            continue
            
        # Perform sieve check for window [p^2+1, p^2+200]
        # We need to divide by all primes q <= sqrt(p)
        sqrt_p = int(math.isqrt(p))
        
        # List of primes to use for sieving
        # We have primes up to 10000. If sqrt_p > 10000, we need more.
        # For p=200000, sqrt_p ~ 447. So 10000 is enough.
        
        active_primes = [q for q in primes if q <= sqrt_p]
        
        # Initialize remainders
        # We only care if any remainder > p.
        # We can stop early if we find one.
        
        # To optimize, we can track the maximum remainder?
        # No, we need to divide.
        
        # Optimization: 
        # Instead of an array of 200 ints, we can just check each number?
        # No, the sieve approach is to divide multiple numbers by q at once.
        
        rem = [p * p + i + 1 for i in range(200)]
        
        for q in active_primes:
            # Find first index j such that rem[j] % q == 0
            # rem[j] = p^2 + j + 1
            # We need p^2 + j + 1 = k * q
            # j = k * q - p^2 - 1
            # Find smallest k such that j >= 0
            # k * q >= p^2 + 1
            # k_min = ceil((p^2 + 1) / q)
            
            k_min = (p * p + 1 + q - 1) // q
            j_start = k_min * q - p * p - 1
            
            # j_start might be negative? 
            # k_min * q >= p^2+1 => j_start >= 0.
            
            # Iterate j from j_start to 199 step q
            for j in range(j_start, 200, q):
                while rem[j] % q == 0:
                    rem[j] //= q
        
        # Check validity
        # All rem[j] must be <= p
        # If any rem[j] > p, fail.
        
        k = 0
        for i in range(200):
            if rem[i] <= p:
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k >= 200:
            break
            
    return {'p': best_p, 'k': best_k}
