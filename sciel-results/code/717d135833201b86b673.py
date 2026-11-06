import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds to be safe

    N = 123457
    
    # Strategy: Exhaustive search over all a in [1, N//2]
    # To make this fast, precompute the Largest Prime Factor (LPF) for all numbers up to N.
    
    # 1. Sieve to compute Largest Prime Factor for each number up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    
    # Initialize lpf[i] = i for primes, but we can just do:
    # For each prime p, for each multiple m of p, set lpf[m] = p.
    # Since we iterate p in increasing order, the last assignment is the largest prime factor.
    
    # First, identify primes or just iterate and check primality?
    # Better: Standard sieve logic adapted for LPF.
    
    # We need to know which numbers are prime to iterate? 
    # Actually, we can just iterate i from 2 to N.
    # If lpf[i] == 0, it means i has no smaller prime factors processed yet? 
    # No, that's for smallest prime factor.
    # For Largest Prime Factor:
    # If we iterate p from 2 to N:
    #   if p is prime:
    #     for j in range(p, N+1, p):
    #       lpf[j] = p
    # This works because if j is a multiple of p, p is a factor. 
    # Since we go p=2, 3, 5..., the largest prime factor will be the last one assigned.
    
    # To check if p is prime, we can use a simple sieve or check lpf[p] == p?
    # If we use the assignment method, lpf[p] will be p when we reach p (if p is prime).
    # But we need to know if p is prime to trigger the inner loop.
    # We can maintain a boolean is_prime array.
    
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    # Standard sieve for primes
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    # Now, compute lpf
    # Iterate through primes
    for p in range(2, N + 1):
        if is_prime[p]:
            for j in range(p, N + 1, p):
                lpf[j] = p
    
    # Edge case: lpf[1] is 0, but P(1)=1.
    lpf[1] = 1
    
    # 2. Iterate a from 1 to N//2
    best_val = float('inf')
    best_pair = [1, N]
    
    half_N = N // 2
    
    for a in range(1, half_N + 1):
        b = N - a
        # P(a) is lpf[a]
        # P(b) is lpf[b]
        p_a = lpf[a]
        p_b = lpf[b]
        
        current_max = p_a if p_a > p_b else p_b
        
        if current_max < best_val:
            best_val = current_max
            best_pair = [a, b]
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_pair
