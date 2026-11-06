def construct():
    answers = []
    MAX_DENOM = 10**12
    
    for n in range(3, 257):
        found = None
        start_x = (n // 4) + 1
        
        # Try x in a reasonable range
        for x in range(start_x, start_x + 2000):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let U = Ay - B, then V = B^2 / U
            # y = (U + B) / A, z = (V + B) / A
            # Need: U divides B^2, U > 0, (U+B) % A == 0, (V+B) % A == 0,
            #       y > 0, z > 0, y != z, all distinct, all <= MAX_DENOM
            
            # Only factor if B is manageable
            if B > 10**7:
                continue
            
            # Factor B
            temp = B
            factors = {}
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    temp //= d
                d += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Generate all divisors of B^2
            divisors = [1]
            for p, e in factors.items():
                new_divs = []
                for dv in divisors:
                    for k in range(2 * e + 1):
                        new_divs.append(dv * (p ** k))
                divisors = new_divs
            
            # For each divisor U of B^2, check conditions
            for U in divisors:
                if U <= 0:
                    continue
                if (B * B) % U != 0:
                    continue
                V = (B * B) // U
                
                if (U + B) % A != 0:
                    continue
                y = (U + B) // A
                if y <= 0:
                    continue
                
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                if z <= 0:
                    continue
                
                if x == y or x == z or y == z:
                    continue
                
                if x > MAX_DENOM or y > MAX_DENOM or z > MAX_DENOM:
                    continue
                
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found is not None:
                break
        
        answers.append(found)
    
    return answers
