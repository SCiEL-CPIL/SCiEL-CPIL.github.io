"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    N = 1000000  # max n
    LO = 10000   # min n
    max_val = N + 1  # we need powerful_part up to N+1
    
    # Log-space accumulation for powerful parts
    log_powerful = np.zeros(max_val + 1, dtype=np.float64)
    
    # Generate primes up to sqrt(max_val) using efficient sieve
    limit = int(math.isqrt(max_val)) + 1
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = b'\x00' * len(is_prime[i*i::i])
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Pre-compute log values for primes
    log_primes = {p: math.log(p) for p in primes}
    
    # For each prime, accumulate log contributions
    for p in primes:
        p_int = p
        log_p = log_primes[p_int]
        pk = p_int * p_int
        k = 2
        while pk <= max_val:
            if k == 2:
                add_val = 2.0 * log_p
            else:
                add_val = log_p
            
            # Vectorized addition to all multiples of pk
            log_powerful[pk::pk] += add_val
            
            pk *= p_int
            k += 1
        
        if time.time() - start_time > deadline - 5:
            break
    
    # Compute consecutive log-powerful sums
    if time.time() - start_time > deadline - 3:
        return LO
    
    # n in [LO, N], so n+1 in [LO+1, N+1]
    consecutive_log = log_powerful[LO:N+1] + log_powerful[LO+1:N+2]
    
    # Find the index of the maximum
    best_idx = int(np.argmax(consecutive_log))
    best_n = LO + best_idx
    
    if best_n < LO or best_n > N:
        best_n = LO
    
    return best_n
# EVOLVE-BLOCK-END
