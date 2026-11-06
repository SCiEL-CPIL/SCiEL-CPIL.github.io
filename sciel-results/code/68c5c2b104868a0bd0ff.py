import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 123457
    
    # Sieve of Eratosthenes to find primes
    # Using standard list for sieve, then convert to numpy if needed, or just stick to lists for sieve and numpy for scan
    is_prime = [True] * N
    is_prime[0] = False
    if N > 1:
        is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N) if is_prime[i]]
    
    # Compute Largest Prime Factor (LPF)
    # lp[i] will store the largest prime factor of i
    lp = [1] * N
    
    for p in primes:
        for j in range(p, N, p):
            lp[j] = p
            
    # Convert to numpy arrays for vectorized scan
    lp_arr = np.array(lp)
    
    # a ranges from 1 to N-1
    a_vals = np.arange(1, N)
    b_vals = N - a_vals
    
    # Get P(a) and P(b)
    pa = lp_arr[a_vals]
    pb = lp_arr[b_vals]
    
    # Calculate max(P(a), P(b)) for each pair
    max_vals = np.maximum(pa, pb)
    
    # Find the minimum value and its index
    min_val = np.min(max_vals)
    min_idx = np.argmin(max_vals)
    
    best_a = int(a_vals[min_idx])
    best_b = int(b_vals[min_idx])
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
