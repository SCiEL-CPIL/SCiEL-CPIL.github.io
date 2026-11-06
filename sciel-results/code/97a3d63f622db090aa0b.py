import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    # Sieve up to 10^7
    # This takes ~1-2s and ~10MB
    sieve_limit = 10**7
    primes = sieve_primes(sieve_limit)
    
    best_k = 0
    best_p = 2
    
    # Precompute prime indices for quick lookup? 
    # We just iterate through primes.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Check p-smoothness of p^2+1 ... p^2+200
        # Using NumPy for vectorized division
        base = p * p
        # Use int64 to avoid overflow, p^2 can be up to 10^14
        vals = np.arange(base + 1, base + 201, dtype=np.int64)
        
        # We need to divide by all primes q <= p
        # Find index of last prime <= p
        # Since primes is sorted, we can use bisect
        import bisect
        idx = bisect.bisect_right(primes, p)
        
        # If p is large, this loop is expensive.
        # Optimization: If p is very large, we might skip or use a different method.
        # But for p up to 10^5, it's fast.
        
        # To speed up, we can stop early if all vals are 1?
        # But we need to know if they ARE 1.
        
        # Efficient division loop
        for q in primes[:idx]:
            # Divide vals by q as much as possible
            # Only process if any val is divisible by q
            # To avoid overhead, just do it?
            # For small q, most vals are divisible.
            
            # Vectorized division
            # We need to repeat until no more divisible
            # This is a while loop inside the for loop.
            
            # Optimization: 
            # If q*q > max(vals), then any remaining factor is prime.
            # But we are dividing by q <= p.
            
            # Let's just do the division.
            # To speed up, we can check `np.any(vals % q == 0)`
            # If false, skip.
            
            # However, for small q, `vals % q` is cheap.
            
            # Let's try a simple loop first.
            while True:
                mask = vals % q == 0
                if not np.any(mask):
                    break
                vals[mask] //= q
                
        # Now check the prefix
        # vals[0] corresponds to p^2+1
        # We need the largest k such that vals[0]...vals[k-1] are all 1
        k = 0
        for v in vals:
            if v == 1:
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
