import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    
    # Precompute largest prime factor for all numbers up to LIMIT
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (LIMIT + 1)
    
    # Sieve-like approach
    # Iterate through potential primes
    for p in range(2, LIMIT + 1):
        if lpf[p] == 0:
            # p is prime
            # Update multiples of p
            # Since we go in increasing order, the last update will be the largest prime factor
            for j in range(p, LIMIT + 1, p):
                lpf[j] = p
                
    # Find the n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = float('inf')
    
    # Check time occasionally? The loop is 1e6 iterations, very fast.
    # Precomputation is ~2.6e6 ops. Total < 1e7 ops.
    # Should take < 1 second in Python.
    
    for n in range(10000, MAX_N + 1):
        val = lpf[n] if lpf[n] > lpf[n+1] else lpf[n+1]
        if val < best_val:
            best_val = val
            best_n = n
            # Early exit if we found a very small value?
            # Theoretical minimum for n>1 is 2? No, n and n+1 coprime.
            # One is even (lpf >= 2), other is odd (lpf >= 3).
            # So min possible is max(2, 3) = 3?
            # Example: n=2, n+1=3. L(2)=3.
            # But n >= 10000.
            # Can we get 3?
            # Need one number to be power of 2, other power of 3.
            # 2^a and 3^b differ by 1.
            # 3^1 - 2^1 = 1 (n=2)
            # 3^2 - 2^2 = 5
            # 3^3 - 2^3 = 19
            # ...
            # Catalan's conjecture (Mihăilescu's theorem) states that the only solution in natural numbers of x^a - y^b = 1 for x, y > 0 and a, b > 1 is 3^2 - 2^3 = 1.
            # So 9 and 8. n=8, n+1=9. L(8) = max(2, 3) = 3.
            # But n >= 10000.
            # Are there other solutions? No, for a,b > 1.
            # If a=1 or b=1, e.g. 2^a - 3 = 1 => 2^a = 4 => a=2 (n=4, L=3).
            # 3^b - 2 = 1 => 3^b = 3 => b=1 (n=2, L=3).
            # So for n >= 10000, the minimum will be larger than 3.
            # Likely 5 or 7.
            
            # We can't easily early exit without knowing the lower bound.
            # But the loop is fast.
            
    return best_n

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    
    lpf = [0] * (LIMIT + 1)
    
    for p in range(2, LIMIT + 1):
        if lpf[p] == 0:
            for j in range(p, LIMIT + 1, p):
                lpf[j] = p
                if time.time() > deadline:
                    # Fallback if somehow slow (shouldn't happen)
                    return 10000
                    
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, MAX_N + 1):
        val = lpf[n] if lpf[n] > lpf[n+1] else lpf[n+1]
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
