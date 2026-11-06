"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Precompute Euler's totient function up to a reasonable bound using a sieve.
    # Then scan from 1 upward to find the smallest n with phi(n) % 120 == 0.
    # The answer is expected to be small, so we can start with a modest limit
    # and expand if needed.
    
    # Start with a limit that's likely sufficient for the smallest such n.
    # phi(143) = phi(11*13) = 10*12 = 120, so n=143 works.
    # Let's check if there's a smaller one by computing phi up to, say, 200 first.
    # Actually, let's just compute up to 1000000 to be safe, but we can do it efficiently.
    
    limit = 1000000
    phi = list(range(limit + 1))
    
    # Sieve for totient
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n >= 1 with phi(n) % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    return -1
# EVOLVE-BLOCK-END
