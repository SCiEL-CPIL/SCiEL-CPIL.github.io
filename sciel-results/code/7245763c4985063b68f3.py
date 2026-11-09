"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We use a sieve to compute phi(n) for all n up to a reasonable limit, then find
    the smallest n where phi(n) % 120 == 0.
    
    We know from number theory that phi(143) = phi(11*13) = 10*12 = 120, so n=143
    is a candidate. Let's verify there's nothing smaller.
    
    Actually, let's just compute it properly with a sieve for correctness.
    """
    import time
    
    deadline = time.time() + 85  # Leave 5s safety margin
    
    # We'll search up to a reasonable limit. Since phi(n) >= 1 and we need phi(n) divisible by 120,
    # and phi(n) <= n, we need n >= 120. Let's search up to 1000000 but we expect to find
    # the answer much earlier.
    
    # Use Euler's totient sieve
    max_n = 1000000
    phi = list(range(max_n + 1))
    
    for i in range(2, max_n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, max_n + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n where phi(n) % 120 == 0
    for n in range(1, max_n + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback: should not reach here
    return 143
# EVOLVE-BLOCK-END
