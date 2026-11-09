"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi(n) for all n up to 1000000.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 1000000
    # Initialize phi[i] = i
    phi = list(range(N + 1))
    
    # Sieve to compute phi
    # For each prime p, multiply phi[j] by (1 - 1/p) for all multiples j of p
    # phi[j] = phi[j] * (p-1) // p
    for i in range(2, int(N**0.5) + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n such that phi[n] % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not happen)
    return 14400

# EVOLVE-BLOCK-END
