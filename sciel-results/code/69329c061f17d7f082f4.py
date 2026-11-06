import math

def construct():
    N = 1000001
    # Sieve of Eratosthenes to find primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # S[m] will hold the square-free part of m (product of p where v_p(m)==1)
    S = [1] * (N + 1)
    
    for p in primes:
        p2 = p * p
        # Iterate through multiples of p
        for m in range(p, N + 1, p):
            if m % p2 != 0:
                S[m] *= p
                
    # P[m] is the powerful part of m
    # P[m] = m // S[m]
    # We want to maximize P[n] * P[n+1] for n in [10000, 1000000]
    
    best_n = 10000
    best_val = -1
    
    for n in range(10000, N):
        pn = n // S[n]
        pn1 = (n + 1) // S[n + 1]
        val = pn * pn1
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
