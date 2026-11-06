"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 123457
    
    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve of Eratosthenes variant to find largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_a = 1
    best_b = N - 1
    best_cost = max(lpf[1], lpf[best_b])
    
    # Search for the split a+b=N that minimizes max(P(a), P(b))
    # We only need to check a up to N//2 because the problem is symmetric
    half_n = N // 2
    for a in range(1, half_n + 1):
        b = N - a
        # P(1) is defined as 1. lpf[1] is 1.
        # For a > 1, lpf[a] is the largest prime factor.
        cost = max(lpf[a], lpf[b])
        if cost < best_cost:
            best_cost = cost
            best_a = a
            best_b = b
            
    return [best_a, best_b]
# EVOLVE-BLOCK-END
