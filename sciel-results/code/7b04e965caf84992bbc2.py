import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for the 120s limit
    
    # Strategy:
    # 1. Segmented sieve for k up to 10^8 to find collisions.
    # 2. Generate specific forms of k (e.g., 2^a 3^b) to find more collisions.
    # 3. Return the M with the most k's.
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Helper to update best
    def update_best(M_val, ks):
        nonlocal best_M, best_k
        if not ks:
            return
        if M_val > 10**18:
            return
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > len(best_k):
            best_M = M_val
            best_k = valid_ks

    # Part 1: Segmented Sieve for k up to 10^8
    # We want to find M = k * sigma(k).
    # We can't store sigma(k) for all k up to 10^8 in a simple list if we want to be safe with memory?
    # 10^8 ints is ~800MB in Python? No, list of ints is much larger.
    # array('Q') or numpy array is better.
    # But we don't need to store all sigma(k). We just need to find collisions.
    # However, to find collisions, we need to compute k*sigma(k) for each k and check if it exists in a dict.
    # The dict will store M -> list of k's.
    # Number of distinct M values is roughly 10^8.
    # Dict with 10^8 entries is too large.
    
    # Alternative:
    # We can just compute k*sigma(k) and store in a dict, but prune the dict if it gets too big?
    # Or, we can just look for small M?
    # If k is large, M = k * sigma(k) is large.
    # Collisions for large M are rare.
    # Collisions for small M require small k.
    # The known best (333312) has k ~ 300-400.
    # So scanning up to 10^8 is likely to find nothing better unless there's a structural reason.
    
    # Let's refine the search range.
    # If we scan k up to 10^6, we cover the known solutions.
    # Let's scan up to 10^7 with a segmented sieve to be safe.
    
    max_k = 10**7
    seg_size = 10**6
    
    # We need sigma(k) for each segment.
    # To compute sigma(k) for k in [L, R], we need prime factors of k.
    # We can precompute smallest prime factor (SPF) for all primes up to sqrt(R).
    # sqrt(10^7) ~ 3162.
    # Primes up to 3162 are few.
    
    # 1. Sieve primes up to sqrt(max_k)
    limit = int(math.isqrt(max_k)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # 2. Segmented sieve for sigma
    # For each segment [L, R], we compute sigma(k) for all k in [L, R].
    # We maintain an array `sigma_arr` of size seg_size, initialized to 1.
    # And an array `temp_arr` of size seg_size, initialized to L, L+1, ... R.
    # For each prime p, we update sigma_arr for multiples of p.
    # Wait, this is O(N log log N) but with a large constant if done naively in Python.
    # Let's use numpy for the segment.
    
    import numpy as np
    
    collisions = {}
    
    for L in range(1, max_k + 1, seg_size):
        R = min(L + seg_size - 1, max_k)
        n = R - L + 1
        
        # temp[i] = L + i
        temp = np.arange(L, R + 1, dtype=np.int64)
        sigma_arr = np.ones(n, dtype=np.int64)
        
        # For each prime p <= sqrt(R)
        for p in primes:
            if p * p > R:
                break
            
            # Find first multiple of p in [L, R]
            start = (L + p - 1) // p * p
            if start < L:
                start += p
            
            # Indices in temp: (start - L) to (R - L) with step p
            # We need to factor out p from temp[i] and update sigma_arr[i]
            
            # Vectorized approach:
            # indices = np.arange(start, R + 1, p)
            # idx = indices - L
            # factors = temp[idx]
            # term = 1
            # while factors % p == 0:
            #    factors //= p
            #    term *= p
            # sigma_arr[idx] *= (term * (p + 1)) // (p - 1)
            # temp[idx] = factors
            
            # This loop inside the prime loop might be slow if not vectorized well.
            # Let's try.
            
            idx = np.arange(start - L, n, p)
            if len(idx) == 0:
                continue
            
            factors = temp[idx].copy()
            term = np.ones(len(idx), dtype=np.int64)
            
            # Factor out p
            # factors //= p**k
            # We can use a loop or log.
            # Since p is small, the power is small.
            mask = factors % p == 0
            while np.any(mask):
                factors[mask] //= p
                term[mask] *= p
                mask = factors % p == 0
            
            # Update sigma
            # sigma(k) = sigma(k/p^a) * (1 + p + ... + p^a)
            # term = p^(a+1) - 1 / (p - 1) ? No, term is sum of powers?
            # My loop:
            # term starts at 1.
            # If factor has p^a, loop runs a+1 times?
            # Initial: factors % p == 0.
            # Iter 1: factors /= p, term *= p. (term = p)
            # Iter 2: factors /= p, term *= p. (term = p^2)
            # ...
            # If a=0, mask is false, term=1. Sum = 1. Correct.
            # If a=1, mask true once. term=p. Sum = 1+p. Correct.
            # Formula: (term * p - 1) // (p - 1)?
            # If term = p^(a+1), sum = (p^(a+1)-1)/(p-1).
            # My term is p^(a+1).
            # So sum = (term - 1) // (p - 1).
            
            sum_divs = (term - 1) // (p - 1)
            
            # But wait, sigma_arr[idx] currently holds the product of contributions from previous primes.
            # We need to multiply by sum_divs.
            # But we also need to ensure we don't double count if p divides temp multiple times?
            # The sieve logic:
            # sigma(k) is multiplicative.
            # If we process prime p, we extract all p from k.
            # The contribution of p^a to sigma(k) is 1 + p + ... + p^a.
            # We multiply sigma_arr by this contribution.
            
            sigma_arr[idx] = sigma_arr[idx] * sum_divs
            temp[idx] = factors
            
        # After processing all primes <= sqrt(R),
        # temp[i] is either 1 or a prime > sqrt(R).
        # If temp[i] > 1, it's a prime factor q.
        # Contribution is 1 + q.
        
        mask = temp > 1
        if np.any(mask):
            sigma_arr[mask] = sigma_arr[mask] * (temp[mask] + 1)
        
        # Now compute M = k * sigma(k)
        ks = np.arange(L, R + 1)
        Ms = ks * sigma_arr
        
        # Check for collisions
        # We can't just put 10^6 entries into a dict every segment if we want to keep it small?
        # Actually, we want to find M that appear multiple times.
        # We can update the global collisions dict.
        
        # To avoid huge memory, we can just check if M is in collisions.
        # But iterating 10^6 items in Python is slow.
        # Let's use numpy to find unique M in this segment?
        # No, we need to match with previous segments.
        
        # Simple loop might be too slow for 10^7 iterations.
        # 10^7 iterations in Python ~ 1s? Maybe 2-3s.
        # 10 segments * 3s = 30s. Acceptable.
        
        for i in range(n):
            M_val = int(Ms[i])
            k_val = int(ks[i])
            
            if M_val > 10**18:
                continue
                
            if M_val in collisions:
                collisions[M_val].append(k_val)
                if len(collisions[M_val]) > len(best_k):
                    update_best(M_val, collisions[M_val])
            else:
                collisions[M_val] = [k_val]
                
        # Time check
        if time.time() > deadline - 10:
            break
            
    # Part 2: Structural search for 2^a 3^b
    # k = 2^a 3^b
    # sigma(k) = (2^(a+1)-1) * (3^(b+1)-1)/2
    # M = 2^a 3^b * (2^(a+1)-1) * (3^(b+1)-1)/2
    # We can generate these M values and check for collisions.
    
    # Let's just rely on the sieve for now, it's more general.
    
    # Ensure types
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}

# EVOLVE-BLOCK-END
