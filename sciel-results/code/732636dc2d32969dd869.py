import time
import math

def construct():
    deadline = time.time() + 85  # Safety margin
    answers = []
    
    # Precompute primes up to 300 for factoring b = nx
    # Max b is approx 256 * 192 = 49152. sqrt(49152) ~ 222.
    limit = 300
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    def get_divisors_from_factors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            pe = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * pe)
                pe *= p
            divs = new_divs
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Iterate x from x_max down to x_min
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # Factor b
            b_factors = factorize(b)
            # Square exponents for b^2
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            
            # Get divisors of b^2
            divs = get_divisors_from_factors(b2_factors)
            
            for u in divs:
                if time.time() > deadline:
                    break
                if found:
                    break
                
                # u = Ay - B => u + B must be divisible by A
                if (u + b) % a != 0:
                    continue
                
                y = (u + b) // a
                if y <= x:
                    continue
                
                v = (b * b) // u
                if (v + b) % a != 0:
                    continue
                
                z = (v + b) // a
                if z <= y:
                    continue
                
                # Distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Bounds
                if z > 10**12 or y > 10**12 or x > 10**12:
                    continue
                
                # Verification
                if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                    found = [x, y, z]
                    break
        
        answers.append(found)
        
    return answers
