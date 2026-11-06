def construct():
    # Use a sieve to compute phi(n) for all n up to 1000000
    # This is O(N log log N) and very fast for N=1e6
    
    limit = 1000000
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i: # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n with phi(n) % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 1 # Fallback, shouldn't happen
