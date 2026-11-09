import time

def construct():
    # Known 3-term solution
    best_solution = [8, 31, 248]
    best_M = 248
    
    # Search for 4-term solutions with M < 248
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a <= b <= c <= d <= M
    # M < 248
    
    # To be safe and fast, we can just check if ANY solution exists with max denom < 248.
    # We iterate M from 1 to 247.
    # However, iterating M and then a,b,c is O(M^4) if we do it naively for each M?
    # No, for each M, we check a,b,c. That's O(M^3) per M? No.
    # For a fixed M, a,b,c are bounded by M.
    # So total work is sum_{M=1}^{247} O(M^3) ~ 247^4 / 4 ~ 10^9. Too slow.
    
    # Better approach:
    # Iterate a, b, c directly, bounded by 247.
    # Calculate d.
    # If d is valid and d <= 247, then we have a solution with max denom <= 247.
    # We want the MINIMUM max denom.
    # So we can collect all valid (a,b,c,d) with max <= 247 and find the one with min max.
    
    # a must be at least 7 (since 1/6 > 5/31)
    # a <= 18 (since 4/a >= 5/31 => a <= 24.8, but actually 1/a < 5/31 => a > 6.2)
    # Actually, if we use 4 terms, a can be larger?
    # 1/a + 1/b + 1/c + 1/d = 5/31.
    # If a=1, 1-5/31 = 26/31. 1/b+1/c+1/d = 26/31.
    # But we want to MINIMIZE max denom.
    # Small a leads to large remainder, which might need large d?
    # Or small a allows smaller b,c,d?
    # Example: 1/2 + ...
    # If we pick small a, the remainder is large, so b,c,d can be small?
    # Let's just search all a,b,c,d <= 247.
    
    min_max_denom = 248
    best_sol = [8, 31, 248]
    
    # a <= b <= c <= d <= 247
    # a >= 1
    # 1/a + 1/b + 1/c + 1/d = 5/31
    
    # Bounds for a:
    # 4/a >= 5/31 => a <= 24
    # 1/a < 5/31 => a > 6
    
    for a in range(7, 25):
        # b >= a
        # 1/b + 1/c + 1/d = 5/31 - 1/a
        rem_a = 5/31 - 1/a
        if rem_a <= 0: continue
        
        # 3/b >= rem_a => b <= 3/rem_a
        max_b = int(3 / rem_a)
        if max_b > 247: max_b = 247
        
        for b in range(a, max_b + 1):
            rem_b = rem_a - 1/b
            if rem_b <= 0: continue
            
            # 2/c >= rem_b => c <= 2/rem_b
            max_c = int(2 / rem_b)
            if max_c > 247: max_c = 247
            
            for c in range(b, max_c + 1):
                rem_c = rem_b - 1/c
                if rem_c <= 0: continue
                
                # 1/d = rem_c
                # d = 1/rem_c
                # Check if d is integer
                # Use fractions for exactness
                # rem_c is float, might have precision issues.
                # Better to use integers.
                
                # Let's stick to integer arithmetic.
                # 1/a + 1/b + 1/c + 1/d = 5/31
                # 31(d(a+b+c) + abc) = 5abc
                # This is hard to solve for d directly without floats or big ints.
                # d = abc / (31(a+b+c) - 5abc) ? No.
                # 1/d = 5/31 - 1/a - 1/b - 1/c
                # 1/d = (5abc - 31bc - 31ac - 31ab) / (31abc)
                # d = 31abc / (5abc - 31bc - 31ac - 31ab)
                
                num = 31 * a * b * c
                den = 5 * a * b * c - 31 * b * c - 31 * a * c - 31 * a * b
                
                if den <= 0:
                    continue
                
                if num % den == 0:
                    d = num // den
                    
                    if d < c:
                        continue
                    if d > 247:
                        continue
                    
                    # Check distinctness
                    if len({a, b, c, d}) == 4:
                        current_max = max(a, b, c, d)
                        if current_max < min_max_denom:
                            min_max_denom = current_max
                            best_sol = [a, b, c, d]
    
    return best_sol
