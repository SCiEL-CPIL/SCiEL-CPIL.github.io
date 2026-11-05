import time
import math

def construct():
    deadline = time.time() + 55.0
    answers = []
    
    # Precompute smallest prime factor for fast factorization
    MAX_B = 256 * 400  # B = n*x, safe upper bound
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_B + 1))
    for i in range(2, int(MAX_B ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Factorize num using precomputed SPF. Returns dict of prime->exponent."""
        if num <= 1:
            return {}
        factors = {}
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors[p] = cnt
        return factors
    
    def get_divisors_of_square(num):
        """Returns sorted list of divisors d of num^2 where d <= num."""
        factors = factorize(num)
        divs = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divs:
                pk = 1
                for k in range(2 * exp + 1):
                    new_divs.append(d * pk)
                    pk *= p
            divs = new_divs
        divs.sort()
        return [d for d in divs if d <= num]
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        x_min = n // 4 + 1
        
        max_x_attempts = 50
        for i in range(max_x_attempts):
            x = x_min + i
            if time.time() > deadline:
                answers.append(None)
                found = None
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # d_start: need y > x => (d+B)/A > x => d > Ax - B = 4x^2 - 2nx
            d_start = 4 * x * x - 2 * n * x + 1
            if d_start < 1:
                d_start = 1
                
            # Get divisors of B^2 that are <= B
            divs = get_divisors_of_square(B)
            
            for d in divs:
                if d < d_start:
                    continue
                if d > B:
                    break
                    
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                d_prime = (B * B) // d
                num_z = d_prime + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if y > x and z > y:
                    found = [x, y, z]
                    break
                    
            if found:
                break
                
        if time.time() > deadline:
            answers.append(None)
        else:
            answers.append(found)
            
    return answers
