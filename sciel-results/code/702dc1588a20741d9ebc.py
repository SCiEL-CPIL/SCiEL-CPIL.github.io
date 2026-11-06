import math

def construct():
    # Precompute primes up to sqrt(1000001) ~ 1000
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    
    def get_powerful_part(m):
        if m < 2:
            return 1
        temp = m
        powerful_part = 1
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                if exp >= 2:
                    powerful_part *= (p ** exp)
        # If temp > 1, it is a prime factor with exponent 1
        # So it is not part of the powerful part
        return powerful_part

    best_n = 10000
    best_val = -1
    
    # Search from top down to leverage pruning
    # n is in [10000, 1000000]
    for n in range(1000000, 9999, -1):
        # Pruning: max possible product for this n and below is n*(n+1)
        # If this is <= best_val, we can stop
        if n * (n + 1) <= best_val:
            break
        
        pn = get_powerful_part(n)
        pn1 = get_powerful_part(n + 1)
        val = pn * pn1
        
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
