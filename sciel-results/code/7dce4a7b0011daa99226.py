import random
import math

def construct():
    # Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    # We want to maximize k. Max possible is 200.
    # Search in high range where collisions are rare.
    # Random search
    best_k = 1
    best_s = 1
    
    # Increase attempts to find k=200
    # 300 attempts should be fast enough and increase probability significantly
    for _ in range(300):
        s = random.randint(10**11, 10**12 - 200)
        
        seen = set()
        k = 0
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
                
    return {'start': best_s, 'length': best_k}
