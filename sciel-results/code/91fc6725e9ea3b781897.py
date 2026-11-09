"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Find smallest n in [1, 1000000] with phi(n) divisible by 120.
    # Use a sieve to compute phi(n) for n up to a limit.
    # The minimum is expected to be small (around 143), so a modest limit suffices.
    
    N = 10000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] = phi[j] // i * (i - 1)
    
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback: should not be reached since minimum is well below 10000
    return 143

# EVOLVE-BLOCK-END
