"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # Safety margin from 90s limit
    
    N = 1000000
    # Initialize phi[i] = i
    phi = list(range(N + 1))
    
    # Sieve for totient
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] = phi[j] // i * (i - 1)
        
        # Check timeout periodically (e.g., every 1000 iterations of outer loop)
        if i % 1000 == 0:
            if time.time() - start_time > timeout:
                # Fallback to a known valid small number if we run out of time
                # But the sieve should finish very fast.
                # If we haven't found the min yet, we can't return a partial sieve result easily.
                # However, the sieve is O(N log log N) which is very fast for 1e6.
                pass

    # Find smallest n with phi[n] divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    return -1
# EVOLVE-BLOCK-END
