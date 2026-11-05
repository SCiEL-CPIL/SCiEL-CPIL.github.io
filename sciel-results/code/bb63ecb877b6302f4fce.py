def construct():
    import time
    
    deadline = time.time() + 82  # 82s deadline with 8s margin
    
    # Target: 5/31
    # We want to minimize the largest denominator in an Egyptian fraction expansion.
    # Strategy:
    # 1. Systematically search for 3-term solutions 1/a + 1/b + 1/c = 5/31 with a < b < c.
    #    For each pair (a, b), c is uniquely determined. We check if c is a valid integer.
    # 2. If no 3-term solution is found with a small max denominator, search for 4-term solutions.
    # 3. Use the Diophantine equation structure for efficiency.
    
    best_solution = None
    best_max = 10000001
    
    # --- Phase 1: Search for 3-term solutions ---
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # Bounds for a:
    # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
    # So a is in [7, 18]
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        
        # Remaining sum: r = 5/31 - 1/a = (5a - 31) / (31a)
        num_r = 5 * a - 31
        den_r = 31 * a
        
        if num_r <= 0:
            continue
        
        # We need 1/b + 1/c = num_r / den_r with b > a and c > b
        # 1/b < num_r/den_r => b > den_r/num_r
        # 1/b > num_r/(2*den_r) => b < 2*den_r/num_r
        # So b is in (den_r/num_r, 2*den_r/num_r)
        # Also b > a
        
        b_min = max(a + 1, den_r // num_r + 1)
        b_max = (2 * den_r) // num_r
        
        # We want to minimize c, so we should check b values that lead to small c.
        # c = den_r * b / (num_r * b - den_r)
        # As b increases from b_min, the denominator (num_r*b - den_r) increases, 
        # so c decreases? Let's check.
        # f(b) = den_r * b / (num_r * b - den_r)
        # f'(b) = [den_r(num_r*b - den_r) - den_r*b*num_r] / (num_r*b - den_r)^2
        #       = [den_r*num_r*b - den_r^2 - den_r*num_r*b] / ...
        #       = -den_r^2 / ... < 0
        # So c decreases as b increases. Thus, to minimize c, we want the largest possible b.
        # However, we must ensure c is an integer and c > b.
        
        # Iterate b from b_max down to b_min to find the smallest c first?
        # Or just iterate and keep the best. Since the range is small, just iterate all.
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            
            # 1/c = num_r/den_r - 1/b = (num_r*b - den_r) / (den_r*b)
            num_c = num_r * b - den_r
            den_c = den_r * b
            
            if num_c <= 0:
                continue
            
            if den_c % num_c != 0:
                continue
            
            c = den_c // num_c
            
            if c <= b:
                continue
            
            if c > 10000000:
                continue
            
            # Check distinctness
            if a == b or b == c or a == c:
                continue
            
            # Verify
            # 1/a + 1/b + 1/c == 5/31
            # (bc + ac + ab) / (abc) == 5/31
            # 31*(bc + ac + ab) == 5*abc
            lhs = 31 * (b * c + a * c + a * b)
            rhs = 5 * a * b * c
            
            if lhs != rhs:
                continue
            
            max_val = max(a, b, c)
            if max_val < best_max:
                best_max = max_val
                best_solution = [a, b, c]
                
                # If we find a very small max, we can return early
                if best_max <= 100:
                    return best_solution
    
    if best_solution is not None:
        return best_solution
    
    # --- Phase 2: Search for 4-term solutions ---
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # a is in [7, 18] as before (roughly, might be slightly different for 4 terms)
    # For 4 terms, 1/a + 1/b + 1/c + 1/d = 5/31
    # 1/a < 5/31 => a >= 7
    # 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
    
    for a in range(7, 25):
        if time.time() > deadline:
            break
        
        num_r1 = 5 * a - 31
        den_r1 = 31 * a
        
        if num_r1 <= 0:
            continue
        
        # 1/b + 1/c + 1/d = num_r1 / den_r1
        # b > a
        # 1/b < num_r1/den_r1 => b > den_r1/num_r1
        # 3/b > num_r1/den_r1 => b < 3*den_r1/num_r1
        
        b_min = max(a + 1, den_r1 // num_r1 + 1)
        b_max = (3 * den_r1) // num_r1
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            
            num_r2 = num_r1 * b - den_r1
            den_r2 = den_r1 * b
            
            if num_r2 <= 0:
                continue
            
            # 1/c + 1/d = num_r2 / den_r2
            # c > b
            # 1/c < num_r2/den_r2 => c > den_r2/num_r2
            # 2/c > num_r2/den_r2 => c < 2*den_r2/num_r2
            
            c_min = max(b + 1, den_r2 // num_r2 + 1)
            c_max = (2 * den_r2) // num_r2
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                
                num_d = num_r2 * c - den_r2
                den_d = den_r2 * c
                
                if num_d <= 0:
                    continue
                
                if den_d % num_d != 0:
                    continue
                
                d = den_d // num_d
                
                if d <= c:
                    continue
                
                if d > 10000000:
                    continue
                
                if len({a, b, c, d}) != 4:
                    continue
                
                # Verify
                # 1/a + 1/b + 1/c + 1/d == 5/31
                # 31*(bcd + acd + abd + abc) == 5*abcd
                lhs = 31 * (b * c * d + a * c * d + a * b * d + a * b * c)
                rhs = 5 * a * b * c * d
                
                if lhs != rhs:
                    continue
                
                max_val = max(a, b, c, d)
                if max_val < best_max:
                    best_max = max_val
                    best_solution = [a, b, c, d]
                    
                    if best_max <= 100:
                        return best_solution
    
    if best_solution is not None:
        return best_solution
    
    # --- Fallback: Greedy method ---
    # This should not be reached if the above searches work correctly,
    # but it's a safe fallback.
    from fractions import Fraction
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
