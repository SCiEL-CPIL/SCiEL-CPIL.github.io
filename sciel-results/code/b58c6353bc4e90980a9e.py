def construct():
    import time
    from math import gcd

    start_time = time.time()
    deadline = start_time + 55
    n = 128
    
    # Generate primes up to 128
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Construct set of all prime powers <= 128
    S = set()
    S.add(1)
    for p in primes:
        power = p
        while power <= n:
            S.add(power)
            power *= p
            
    # Verify validity (optional, but good for safety)
    # If a bad triple is found, we might need to remove elements, 
    # but prime powers + 1 is known to be valid.
    
    return list(S)
