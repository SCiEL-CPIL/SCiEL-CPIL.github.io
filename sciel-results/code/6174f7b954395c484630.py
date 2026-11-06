import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85
    
    def elapsed():
        return time.time() - t0
    
    MAXN = 1_000_000
    MAXN_PLUS = MAXN + 1
    
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, MAXN + 1) if is_prime[i]]
    
    pp = np.ones(MAXN_PLUS, dtype=np.int64)
    
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        p_sq = p * p
        if p_sq > MAXN:
            break
        
        p_power = p_sq
        while p_power <= MAXN:
            next_power = p_power * p
            max_k = MAXN // p_power
            
            if next_power <= MAXN:
                # Generate k values not divisible by p
                # k = 1, 2, ..., max_k
                # Exclude k where k % p == 0
                # This is equivalent to: for each block of p consecutive k values,
                # take the first p-1
                n_full_blocks = max_k // p
                remainder = max_k % p
                n_valid = n_full_blocks * (p - 1) + min(remainder, p - 1)
                
                # Generate valid k values
                ks = np.empty(n_valid, dtype=np.int64)
                # For each full block, add p-1 values
                # Block j (0-indexed): k = j*p + 1, j*p + 2, ..., j*p + (p-1)
                # Last partial block: k = n_full_blocks*p + 1, ..., n_full_blocks*p + min(remainder, p-1)
                
                # Use a more efficient method
                all_ks = np.arange(1, max_k + 1, dtype=np.int64)
                mask = (all_ks % p) != 0
                ks = all_ks[mask]
                indices = ks * p_power
            else:
                indices = np.arange(1, max_k + 1, dtype=np.int64) * p_power
            
            if len(indices) > 0:
                pp[indices] *= p_power
            
            p_power = next_power

    if elapsed() > DEADLINE:
        return 10000
    
    products = pp[10000:MAXN] * pp[10001:MAXN_PLUS]
    best_idx = np.argmax(products)
    best_n = 10000 + best_idx
    
    return int(best_n)
