"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi(n) for all n up to a limit,
    then finds the smallest n satisfying the condition.
    """
    # We need phi(n) % 120 == 0
    # 120 = 2^3 * 3 * 5
    
    # Use a sieve to compute phi(n) for n up to N
    N = 100000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n such that phi[n] % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should not reach here)
    return 14400
# EVOLVE-BLOCK-END
