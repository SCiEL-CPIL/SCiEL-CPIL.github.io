import time
import math
from math import isqrt

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 90s limit (evaluator kills at 90s)
    results = [None] * 254  # n=3..256, index 0 corresponds to n=3
    
    # Precompute smallest prime factor up to a reasonable bound
    # B = n*x, x up to ~3n/4, so B up to ~3*256^2/4 = 49152
    # But we may need larger B for some x values, let's be generous
    max_spf = 200000
    spf = list(range(max_spf + 1))
    for i in range(2, isqrt(max_spf) + 1):
        if spf[i] == i:
            for j in range(i * i, max_spf + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Get prime factorization of num as dict {prime: exponent} using SPF."""
        if num <= 1:
            return {}
        factors = {}
        while num > 1:
            if num <= max_spf:
                p = spf[num]
            else:
                # Fallback for large numbers: trial division
                p = None
                for d in range(2, isqrt(num) + 1):
                    if num % d == 0:
                        p = d
                        break
                if p is None:
                    p = num
            count = 0
            while num % p == 0:
                num //= p
                count += 1
            factors[p] = count
        return factors
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from prime factorization dict."""
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return sorted(divisors)
    
    def find_solution_for_n(n):
        """Find distinct positive integers x,y,z with 1/x+1/y+1/z=4/n."""
        # Strategy: iterate x, then use divisor method for y,z
        # x must satisfy: 1/x < 4/n => x > n/4
        # and 1/x + 1/y + 1/z < 3/x < 4/n => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values that make A = 4x - n small (simpler cases first)
        # But also try x near n/2 for symmetry
        x_candidates = list(range(x_min, x_max + 1))
        x_candidates.sort(key=lambda x: abs(4*x - 2*n))  # prefer x near n/2
        
        for x in x_candidates:
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Need 1/y + 1/z = A/B
            # (Ay - B)(Az - B) = B^2
            # u = Ay - B, v = Az - B, u*v = B^2
            # y = (u+B)/A, z = (v+B)/A
            # u,v positive divisors of B^2 with u*v = B^2
            
            # Factorize B
            b_factors = factorize(B)
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            
            # Get divisors of B^2, but only need those up to B (since u*v=B^2, we can assume u<=v)
            divs = get_divisors_from_factors(b2_factors)
            
            for u in divs:
                if u > B:
                    break  # since divs are sorted
                v = B * B // u
                if v * u != B * B:
                    continue
                
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                if y <= 0:
                    continue
                
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= 0:
                    continue
                
                if y > 10**12 or z > 10**12:
                    continue
                
                if x == y or x == z or y == z:
                    continue
                
                # Verify
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return sorted([x, y, z])
        
        return None
    
    # First pass: try all n
    for idx in range(254):
        if time.time() > deadline:
            break
        n = idx + 3
        sol = find_solution_for_n(n)
        results[idx] = sol
    
    # Second pass: retry unsolved with more aggressive search if time permits
    if time.time() < deadline - 5.0:
        for idx in range(254):
            if time.time() > deadline:
                break
            if results[idx] is not None:
                continue
            n = idx + 3
            # Try a different x ordering
            x_min = n // 4 + 1
            x_max = (3 * n) // 4
            x_candidates = list(range(x_min, x_max + 1))
            x_candidates.sort(key=lambda x: abs(4*x - n))  # prefer small A
            
            found = None
            for x in x_candidates:
                if time.time() > deadline:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                b_factors = factorize(B)
                b2_factors = {p: 2*e for p, e in b_factors.items()}
                divs = get_divisors_from_factors(b2_factors)
                
                for u in divs:
                    if u > B:
                        break
                    v = B * B // u
                    if v * u != B * B:
                        continue
                    
                    num_y = u + B
                    if num_y % A != 0:
                        continue
                    y = num_y // A
                    if y <= 0:
                        continue
                    
                    num_z = v + B
                    if num_z % A != 0:
                        continue
                    z = num_z // A
                    if z <= 0:
                        continue
                    
                    if y > 10**12 or z > 10**12:
                        continue
                    
                    if x == y or x == z or y == z:
                        continue
                    
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = sorted([x, y, z])
                        break
                
                if found:
                    break
            
            if found:
                results[idx] = found
    
    return results
