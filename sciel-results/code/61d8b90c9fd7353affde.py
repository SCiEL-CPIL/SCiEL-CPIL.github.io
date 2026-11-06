"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N_MAX = 1000000
    N_LIMIT = N_MAX + 1  # We need n+1 up to 1000001
    
    # 1. Sieve of Eratosthenes to find primes up to N_LIMIT
    is_prime = np.ones(N_LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    
    # 2. Precompute powerful parts using vectorized NumPy
    # val[x] will store P(x) = product of p^e for each prime p with e>=2 in factorization of x
    val = np.ones(N_LIMIT + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N_LIMIT:
            break
        
        # Iterate through powers p^k for k >= 2
        power = p2
        next_power = power * p
        
        while power <= N_LIMIT:
            # For each multiple m of 'power'
            # Check if exponent of p in m is exactly log_p(power)
            # i.e., m is divisible by power but not by next_power
            
            if next_power > N_LIMIT:
                # All multiples of power have exactly this exponent
                indices = np.arange(power, N_LIMIT + 1, power)
                val[indices] *= power
            else:
                # Multiples of power that are NOT multiples of next_power
                indices = np.arange(power, N_LIMIT + 1, power)
                mask = (indices % next_power) != 0
                val[indices[mask]] *= power
            
            power = next_power
            next_power *= p
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # 3. Find best n in [10000, 1000000] maximizing val[n] * val[n+1]
    best_score = -1
    best_n = 10000
    
    # Vectorized scan for the best n
    scores = val[10000:N_MAX + 1] * val[10001:N_MAX + 2]
    
    # Find the index of the maximum score
    max_idx = np.argmax(scores)
    best_n = 10000 + max_idx
    best_score = scores[max_idx]
    
    return best_n

# EVOLVE-BLOCK-END
