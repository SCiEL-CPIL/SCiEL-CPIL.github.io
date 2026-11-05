"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 1000000
    # Initialize phi array
    phi = list(range(N + 1))
    
    # Sieve of Eratosthenes style for phi
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 1 # Fallback, though 143 is guaranteed to be found

# EVOLVE-BLOCK-END
