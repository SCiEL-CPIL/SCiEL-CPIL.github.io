import math
import time
import random

_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in _MR_BASES:
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

def sieve(limit):
    if limit < 2:
        return []
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_p = 0
    best_k = 0
    
    # Generate primes up to 10^6 for candidate p values
    # Also need primes up to 10^6 for the sieving step
    sieve_limit = 10**6
    primes = sieve(sieve_limit)
    
    # For each candidate prime p, use segmented sieve to check
    # if p^2+1, ..., p^2+k are all p-smooth
    
    for p in primes:
        if time.time() > deadline:
            break
        
        p2 = p * p
        
        # Create segment: values p^2+1 to p^2+200
        # We'll divide out all prime factors <= p
        # If all become 1, then all are p-smooth
        
        # For efficiency, we only need to check up to best_k+1 first
        # If p^2+j is not p-smooth for j <= best_k, skip
        # Otherwise, extend
        
        # First, quick check: is p^2+1 p-smooth?
        # Use segmented sieve for the full range up to 200
        # But to save time, do it incrementally
        
        # Initialize: for each i in 1..200, rem[i] = p^2 + i
        # For each prime q <= p, find i where q | (p^2+i), divide rem[i] by q repeatedly
        
        # Optimization: only process primes q where q <= p and q <= p^2+200
        # (always true for q <= p since p^2+200 > p for p >= 2)
        
        # For each prime q, the residue r = (-p^2) mod q
        # i values: r, r+q, r+2q, ... within [1, 200]
        
        # To avoid allocating 200-element arrays repeatedly, use a list
        # But we need to be fast
        
        # Let's do a quick preliminary check for small k first
        # Check if p^2+1 is p-smooth using direct factorization (fast for one number)
        # This filters out most candidates quickly
        
        # Actually, the segmented sieve is O(pi(p) * 200/q_avg) which is fast
        # Let's just do it
        
        # rem[i] for i=0..199 represents p^2+(i+1)
        # Initialize
        rem = [p2 + i + 1 for i in range(200)]
        
        # For each prime q <= p, divide out factors of q
        for q in primes:
            if q > p:
                break
            
            # Find i in [1,200] where p^2+i ≡ 0 (mod q)
            # i ≡ -p^2 (mod q)
            r = (-p2) % q
            if r == 0:
                r = q  # i=q, 2q, ... (i=0 not in range)
            
            # i values: r, r+q, r+2q, ... <= 200
            # But i must be in [1, 200]
            # r is in [1, q], so if r > 200, no values
            if r > 200:
                continue
            
            for i in range(r - 1, 200, q):  # i-1 index, i is 1-based
                # rem[i-1] = p^2 + i
                # Divide out all factors of q
                while rem[i - 1] % q == 0:
                    rem[i - 1] //= q
        
        # Now check: find largest k such that rem[0..k-1] are all 1
        k = 0
        for i in range(200):
            if rem[i] == 1:
                k = i + 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Fallback
    if best_k == 0:
        for p in [7, 11, 13, 17, 19, 23, 29, 31]:
            p2 = p * p
            k = 0
            while k < 200:
                n = p2 + k + 1
                # Check if n is p-smooth
                temp = n
                for q in primes:
                    if q > p:
                        break
                    while temp % q == 0:
                        temp //= q
                if temp == 1:
                    k += 1
                else:
                    break
            if k > 0:
                best_k = k
                best_p = p
                break
        if best_k == 0:
            best_p = 7
            best_k = 1
    
    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
