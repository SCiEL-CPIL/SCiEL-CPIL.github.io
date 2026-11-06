import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MAX_VAL = 1000000
    N_MIN_VAL = 10000
    
    # 1. Get primes up to sqrt(10^6) = 1000
    limit = 1000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array([i for i, is_prime in enumerate(sieve) if is_prime])
    
    # 2. Initialize powerful part array
    # Using int64 to be safe, though 32-bit might suffice for factors, 
    # the product of factors could be large? 
    # Max powerful part is n itself (if n is powerful). n <= 10^6.
    # So int32 is safe for pp. But let's use int64 for safety in intermediate steps if any.
    # Actually, pp[m] <= m <= 10^6. So int32 is enough.
    pp = np.ones(N_MAX_VAL + 1, dtype=np.int64)
    
    # 3. Sieve to compute powerful parts
    # For each prime p, for each power p^k (k>=2), 
    # multiply pp[m] by p^k for m where v_p(m) == k.
    
    for p in primes:
        p_int = int(p)
        p_k = p_int * p_int
        k = 2
        while p_k <= N_MAX_VAL:
            # p_k is p^k
            # We need to update indices m such that m is multiple of p_k 
            # but NOT multiple of p^(k+1).
            
            # Indices: p_k, 2*p_k, 3*p_k, ...
            # Let m = i * p_k.
            # Condition: i is not divisible by p.
            
            # Create array of indices
            # np.arange(start, stop, step)
            indices = np.arange(p_k, N_MAX_VAL + 1, p_k)
            
            # Calculate i = index // p_k
            # This is equivalent to 1, 2, 3, ...
            # We can avoid creating a separate array for i by using modulo on indices?
            # No, indices % (p * p_k) != 0 is the condition for "not divisible by p^(k+1)"
            # Because if m is divisible by p^(k+1), then m % (p^(k+1)) == 0.
            # p^(k+1) = p * p_k.
            
            next_p_k = p_k * p_int
            mask = (indices % next_p_k) != 0
            
            # Update pp
            pp[indices[mask]] *= p_k
            
            p_k = next_p_k
            k += 1
            
            if time.time() > deadline - 5:
                # Fallback if something goes wrong with time, though this should be fast
                return N_MIN_VAL

    if time.time() > deadline - 5:
        return N_MIN_VAL

    # 4. Find the best n
    # We want to maximize pp[n] * pp[n+1] for n in [10000, 999999]
    # pp is 0-indexed, so pp[10000] corresponds to n=10000.
    # We need n and n+1, so n can go up to 999999.
    
    # Slice for n: 10000 to 999999 (inclusive)
    # Slice for n+1: 10001 to 1000000 (inclusive)
    
    left = pp[N_MIN_VAL:N_MAX_VAL]
    right = pp[N_MIN_VAL + 1:N_MAX_VAL + 1]
    
    products = left * right
    
    # Find index of max product
    max_idx = np.argmax(products)
    
    best_n = N_MIN_VAL + max_idx
    
    return int(best_n)
