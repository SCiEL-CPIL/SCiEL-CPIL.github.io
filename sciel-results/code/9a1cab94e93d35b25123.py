"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    N = 512
    # The set of all divisors of a number K <= N is a valid set,
    # because for any a, b | K, lcm(a, b) | K, so lcm(a, b) <= K <= N.
    # We choose K = 480, which has 24 divisors, a high count for K <= 512.
    # Divisors of 480:
    # 480 = 2^5 * 3 * 5
    # Count = (5+1)*(1+1)*(1+1) = 24.
    
    K = 480
    S = set()
    for i in range(1, K + 1):
        if K % i == 0:
            S.add(i)
            
    return sorted(S)
# EVOLVE-BLOCK-END
