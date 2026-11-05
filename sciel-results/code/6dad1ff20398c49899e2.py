import time
import math

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit
    results = []
    
    # Cache for factorizations
    factor_cache = {}
    
    def factorize(num):
        if num in factor_cache:
            return factor_cache[num]
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        factor_cache[num] = factors
        return factors
    
    def get_divisors_from_factors(factors):
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors
    
    for n in range(3, 257):
        if time.time() > deadline:
            results.append(None)
            continue
        
        found = None
        # x must satisfy n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B
            # u * v = B^2
            # y = (u + B) / A, z = (v + B) / A
            
            # Factorize B to get divisors of B^2
            b_factors = factorize(B)
            b_sq_factors = {p: 2 * e for p, e in b_factors.items()}
            
            divs = get_divisors_from_factors(b_sq_factors)
            
            for u in divs:
                # Check if y is integer
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                if y <= 0:
                    continue
                
                # Calculate v
                v = (B * B) // u
                
                # Check if z is integer
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= 0:
                    continue
                
                # Check distinctness and bounds
                if y == z:
                    continue
                if y == x or z == x:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify: 1/x + 1/y + 1/z = 4/n
                # Cross multiply: n*(yz + xz + xy) == 4*xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        results.append(found)
    
    return results
