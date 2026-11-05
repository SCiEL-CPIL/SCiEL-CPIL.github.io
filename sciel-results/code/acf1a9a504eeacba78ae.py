import math

def construct():
        n = 255
        # Find prime factors of n
        factors = []
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.append(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        
        # Unique prime factors
        primes = list(set(factors))
        
        best_set = []
        max_size = 0
        
        for p in primes:
            # Generate multiples of p in [1, n]
            count = n // p
            if count > max_size:
                max_size = count
                best_set = list(range(p, n + 1, p))
                
        return best_set
