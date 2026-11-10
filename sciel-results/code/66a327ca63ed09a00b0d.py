import math

def construct():
    N = 256
    # Sieve of Eratosthenes to find primes up to 256
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    selected = [1]
    for p in primes:
        max_power = 1
        current = 1
        while current * p <= N:
            current *= p
            max_power = current
        selected.append(max_power)
        
    return selected
