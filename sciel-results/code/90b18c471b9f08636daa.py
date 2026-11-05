import time

# EVOLVE-BLOCK-START

def largest_prime_factor_exact(n):
    """Compute largest prime factor of n via exact trial division."""
    if n <= 1:
        return 1
    lpf = 1
    # Trial divide by 2
    while n % 2 == 0:
        lpf = 2
        n //= 2
    # Trial divide by odd numbers up to sqrt(n)
    f = 3
    while f * f <= n:
        while n % f == 0:
            lpf = f
            n //= f
        f += 2
    if n > 1:
        lpf = n
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    
    N = 1000000
    N_ext = 1000001  # Need lpf up to 1000001 for n=1000000
    
    # 1. Compute largest prime factor for all numbers up to N_ext
    # Optimized sieve: iterate primes in increasing order, update multiples
    lpf = [0] * (N_ext + 1)
    
    for i in range(2, N_ext + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N_ext + 1, i):
                lpf[j] = i
    
    # 2. Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, N + 1):
        if time.time() > deadline:
            break
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        current_val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
    
    # 3. Validate the result with exact trial division
    # This ensures the returned integer strictly minimizes the metric
    if time.time() < deadline - 2.0:  # Leave 2s more for safety
        val_n = largest_prime_factor_exact(best_n)
        val_np1 = largest_prime_factor_exact(best_n + 1)
        exact_val = max(val_n, val_np1)
        
        # If there's a discrepancy, do a local search around best_n
        if exact_val != best_val:
            # Local search in a window around best_n
            window = 100
            for n in range(max(10000, best_n - window), min(N, best_n + window) + 1):
                if time.time() > deadline - 1.0:
                    break
                v1 = largest_prime_factor_exact(n)
                v2 = largest_prime_factor_exact(n + 1)
                v = max(v1, v2)
                if v < exact_val:
                    exact_val = v
                    best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
