def construct():
    """
    Find minimum n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute totients for small n.
    """
    limit = 200
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 143
