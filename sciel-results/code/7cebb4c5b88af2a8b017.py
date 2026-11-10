import time
import math

def construct():
    MAX_DENOM = 10**12
    start_time = time.time()
    TIMEOUT = 50.0  # Safety margin
    
    # Precompute factors for small numbers to speed up B = n*x factorization
    # Max n is 256, max x is roughly 3*256/4 = 192
    max_val = 300
    factors_cache = [None] * (max_val + 1)
    factors_cache[1] = {}
    for i in range(2, max_val + 1):
        if factors_cache[i] is None:
            factors_cache[i] = {}
            num = i
            d = 2
            while d * d <= num:
                while num % d == 0:
                    factors_cache[i][d] = factors_cache[i].get(d, 0) + 1
                    num //= d
                d += 1
            if num > 1:
                factors_cache[i][num] = factors_cache[i].get(num, 0) + 1

    def get_divisors_from_factors(factors):
        divisors = [1]
        for p, e in factors.items():
            current_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divisors:
                    current_divs.append(d * power)
                power *= p
            divisors = current_divs
        return divisors

    answers = []
    for n in range(3, 257):
        if time.time() - start_time > TIMEOUT:
            answers.append(None)
            continue
            
        found = None
        # x must be > n/4 and < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # Factorize B
            # B = n * x
            # Combine factors of n and x
            b_factors = {}
            if n <= max_val:
                for p, e in factors_cache[n].items():
                    b_factors[p] = e
            else:
                # Fallback factorization (should not happen for n<=256)
                num = n
                d = 2
                while d * d <= num:
                    while num % d == 0:
                        b_factors[d] = b_factors.get(d, 0) + 1
                        num //= d
                    d += 1
                if num > 1:
                    b_factors[num] = b_factors.get(num, 0) + 1
            
            if x <= max_val:
                for p, e in factors_cache[x].items():
                    b_factors[p] = b_factors.get(p, 0) + e
            else:
                num = x
                d = 2
                while d * d <= num:
                    while num % d == 0:
                        b_factors[d] = b_factors.get(d, 0) + 1
                        num //= d
                    d += 1
                if num > 1:
                    b_factors[num] = b_factors.get(num, 0) + 1
            
            # Factors of B^2
            b2_factors = {}
            for p, e in b_factors.items():
                b2_factors[p] = 2 * e
            
            # Generate divisors of B^2
            divs = get_divisors_from_factors(b2_factors)
            
            B_sq = B * B
            
            for u in divs:
                if u >= B:
                    continue # We need u < v, so u < B
                
                # Check u + B is divisible by A
                if (u + B) % A != 0:
                    continue
                
                v = B_sq // u
                
                # Check v + B is divisible by A
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness and order
                if x < y < z:
                    if z <= MAX_DENOM:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
