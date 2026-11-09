# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = 60.0  # safety margin before 90s kill
    
    N = 1000000
    # Initialize phi with identity
    phi = list(range(N + 1))
    
    # Sieve to compute phi
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find smallest n with phi[n] % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not happen)
    return 143
# EVOLVE-BLOCK-END
