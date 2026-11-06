import math
import time

def construct():
    answers = []
    deadline = time.time() + 45  # Safety margin for 90s limit, leave time for overhead
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must be > n/4 for A = 4x-n to be positive
        # Start from smallest valid x
        min_x = n // 4 + 1
        # Try x in a reasonable range. Solutions are typically found for small x.
        # Upper bound for x: if x is very large, y and z become small, but we need distinctness.
        # Let's try up to n * 10 or similar. 
        # Actually, for 1/x + 1/y + 1/z = 4/n, if x is large, 1/x is small.
        # 1/y + 1/z approx 4/n. y, z approx n/2.
        # So x doesn't need to be huge.
        
        for x in range(min_x, min_x + 500): 
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # We need (Ay-B)(Az-B) = B^2
            # Let u = Ay-B, v = Az-B. u*v = B^2.
            # u = A*y - B => A*y = u + B => y = (u+B)/A
            # We need u | B^2 and (u+B) % A == 0.
            
            # Find divisors of B^2
            # Factorize B
            b_factors = {}
            temp_b = B
            d = 2
            while d * d <= temp_b:
                while temp_b % d == 0:
                    b_factors[d] = b_factors.get(d, 0) + 1
                    temp_b //= d
                d += 1
            if temp_b > 1:
                b_factors[temp_b] = b_factors.get(temp_b, 0) + 1
            
            # Generate divisors of B^2
            # Exponents in B^2 are 2 * e_i
            divs = [1]
            for p, e in b_factors.items():
                new_divs = []
                exp_max = 2 * e
                for d_val in divs:
                    for k in range(exp_max + 1):
                        new_divs.append(d_val * (p ** k))
                divs = new_divs
            
            # Check each divisor u
            for u in divs:
                # u must be such that y is integer
                if (u + B) % A != 0:
                    continue
                
                y = (u + B) // A
                if y <= 0:
                    continue
                
                # v = B^2 / u
                v = (B * B) // u
                # Check if v gives integer z
                if (v + B) % A != 0:
                    continue
                
                z = (v + B) // A
                if z <= 0:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify solution
                # 1/x + 1/y + 1/z == 4/n
                # (yz + xz + xy) == 4xyz / n
                # n(yz + xz + xy) == 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    # Ensure positive integers (already checked >0)
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
