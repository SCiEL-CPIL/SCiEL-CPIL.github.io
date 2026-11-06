"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # We need powerful_part for n and n+1 where n is in [10000, 1000000]
    # So max index needed is 1000001
    MAX_VAL = 1000001
    powerful_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Generate primes up to sqrt(MAX_VAL)
    limit = int(np.sqrt(MAX_VAL)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    for p in primes:
        p2 = p * p
        if p2 > MAX_VAL:
            break
        q = p2
        while q <= MAX_VAL:
            indices = np.arange(q, MAX_VAL + 1, q)
            next_q = q * p
            if next_q > MAX_VAL:
                powerful_part[indices] *= q
            else:
                mask = (indices % next_q) != 0
                if np.any(mask):
                    powerful_part[indices[mask]] *= q
            q = next_q
    
    # Calculate products for n in [10000, 1000000]
    # n ranges from 10000 to 1000000
    # n+1 ranges from 10001 to 1000001
    products = powerful_part[10000:1000001] * powerful_part[10001:1000002]
    max_idx = np.argmax(products) + 10000
    
    return int(max_idx)

# EVOLVE-BLOCK-END
