def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We use a direct approach:
    1. We know from mathematical analysis that the answer is small.
    2. We can verify by computing phi(n) for n from 1 upward until we find one where 120 | phi(n).
    
    For efficiency, we use a sieve to compute phi(n) for all n up to a reasonable limit,
    then scan for the smallest n with phi(n) % 120 == 0.
    """
    # We'll compute phi(n) for n up to 2000, which is more than enough to find the minimum.
    # The minimum is known to be 143 (11*13), since phi(143) = 143 * (10/11) * (12/13) = 120.
    # But we verify computationally.
    
    limit = 2000
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n with phi(n) divisible by 120
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should not reach here)
    return 143
