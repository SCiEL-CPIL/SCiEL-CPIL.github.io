def construct():
    from fractions import Fraction
    import time
    
    target = Fraction(5, 31)
    best_solution = [8, 31, 248]
    best_max = 248
    
    def try_update(solution):
        nonlocal best_solution, best_max
        m = max(solution)
        if m < best_max:
            best_max = m
            best_solution = list(solution)

    # 3-term analytic search
    # 1/a + 1/b + 1/c = 5/31
    # a in [7, 18]
    for a in range(7, 19):
        n = 5 * a - 31
        d = 31 * a
        if n <= 0:
            continue
        K = d * d
        # X = n*b - d
        # X must divide K
        # b = (X + d) / n
        # X < sqrt(K)
        # b > a => X > n*a - d
        min_X = n * a - d
        sqrt_K = int(K ** 0.5)
        
        # Iterate divisors of K
        # To be efficient, we can just iterate potential X values that are divisors
        # But finding divisors of K up to sqrt(K) is fast for small K
        # K is at most (31*18)^2 ~ 3000^2 ~ 10^7
        # So we can just loop X from 1 to sqrt_K and check if K % X == 0
        
        for X in range(max(1, min_X + 1), sqrt_K + 1):
            if K % X != 0:
                continue
            if (X + d) % n != 0:
                continue
            b = (X + d) // n
            if b <= a:
                continue
            Y = K // X
            if (Y + d) % n != 0:
                continue
            c = (Y + d) // n
            if c <= b:
                continue
            if c > 10000000:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                try_update([a, b, c])

    # 4-term search, looking for max < best_max
    # 1/a + 1/b + 1/c + 1/d = 5/31
    for a in range(7, 19):
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # 1/b + 1/c + 1/d = r1
        # b > a
        # 3/b >= r1 => b <= 3/r1
        b_max = int(3 / r1)
        if b_max > 10000000:
            b_max = 10000000
            
        for b in range(a + 1, b_max + 1):
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            # 1/c + 1/d = r2
            # c > b
            # 2/c >= r2 => c <= 2/r2
            c_max = int(2 / r2)
            if c_max > 10000000:
                c_max = 10000000
            
            for c in range(b + 1, c_max + 1):
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                # 1/d = r3 => d = 1/r3
                if r3.denominator == 1:
                    d = r3.numerator
                elif r3.numerator == 1:
                    d = r3.denominator
                else:
                    continue
                
                if d <= c:
                    continue
                if d > 10000000:
                    continue
                
                # Check if this improves the max denominator
                if d < best_max:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                        try_update([a, b, c, d])

    return best_solution
