import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need P(n+1) for n=N, so up to N+1
    
    # 1. Generate primes up to sqrt(MAX_VAL) using simple sieve
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Initialize powerful part array (use int64 to avoid overflow issues)
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # 3. Compute powerful parts using vectorized sieve
    # For each prime p, for each power q = p^e (e >= 2):
    #   Mark multiples of q that are NOT multiples of q*p by multiplying by q
    
    for p in primes:
        p = int(p)
        q = p * p
        while q <= MAX_VAL:
            next_q = q * p
            
            if time.time() > deadline - 5:
                break
            
            # Indices: q, 2q, 3q, ..., max_k * q
            # We want those where (index // q) % p != 0, i.e., index is not a multiple of next_q
            # Equivalent: index % next_q != 0
            
            # Generate all multiples of q up to MAX_VAL
            max_k = MAX_VAL // q
            if max_k > 0:
                # Vectorized: indices are q, 2q, ..., max_k*q
                # Filter: keep those where (k % p) != 0, equivalently index % next_q != 0
                # Use step arithmetic to avoid creating full array then masking
                # For offsets 1, 2, ..., p-1 (mod p), take indices q*offset, q*(offset+p), ...
                # But simpler: just use np.arange and mask
                idx = np.arange(q, MAX_VAL + 1, q, dtype=np.int64)
                # Mask: idx is not a multiple of next_q
                # idx // q gives k, we want k % p != 0
                # Equivalent: idx % next_q != 0
                mask = (idx % next_q) != 0
                if np.any(mask):
                    P[idx[mask]] *= q
            
            q = next_q
            
        if time.time() > deadline - 5:
            break
    
    # 4. Find best n in [10000, 1000000]
    # Convert to Python list for faster scalar access, or use NumPy
    # P is int64, products can be large but within int64 range for our purposes
    # Actually, let's just iterate in Python for the final search — it's only 990001 iterations
    
    best_n = 10000
    best_val = int(P[10000]) * int(P[10001])
    
    # Use a local variable for speed
    P_list = P.tolist()
    
    for n in range(10000, N + 1):
        val = P_list[n] * P_list[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline - 2:
            break
    
    return best_n
