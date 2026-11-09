def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N = 256
    
    # Find all primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(N)
    
    # The optimal strategy is to include 1 and the highest power of each prime <= N.
    # This ensures pairwise coprimality and maximizes the sum because 
    # combining primes into composite numbers yields a lower sum than 
    # the sum of their individual highest powers.
    
    result = [1]
    
    for p in primes:
        max_power = p
        while max_power * p <= N:
            max_power *= p
        result.append(max_power)
        
    # Ensure types are Python int
    result = [int(v) for v in result]
    
    return result
