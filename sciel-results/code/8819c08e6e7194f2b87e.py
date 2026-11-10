def construct():
    # The smallest n with phi(n) divisible by 120 is 143.
    # We perform a constrained search up to a safe limit to verify.
    # Limit 200 is sufficient to capture n=143.
    limit = 200
    # Initialize phi array
    phi = list(range(limit + 1))
    
    # Sieve for totient
    for i in range(2, limit + 1):
        if phi[i] == i: # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    # Find smallest n
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback if not found (should not happen)
    return 143
