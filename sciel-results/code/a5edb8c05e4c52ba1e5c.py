def construct():
    import time
    from fractions import Fraction
    from math import isqrt
    from functools import lru_cache

    start_time = time.time()
    # Hard deadline: 85s to be safe against 90s kill
    deadline = start_time + 85.0

    target = Fraction(5, 31)
    best_solution = None
    best_max = float('inf')

    @lru_cache(maxsize=None)
    def factorize(n):
        if n <= 1:
            return ()
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return tuple(sorted(factors.items()))

    def get_divisors(factors):
        divs = [1]
        for p, exp in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(exp):
                    val *= p
                    new_divs.append(val)
            divs += new_divs
        return divs

    def find_2_term_solutions(rem, min_den, max_den_limit):
        """
        Find pairs (c, d) with min_den < c < d <= max_den_limit
        such that 1/c + 1/d = rem.
        Returns list of (c, d) or empty list.
        """
        num, den = rem.numerator, rem.denominator
        # (num*c - den)(num*d - den) = den^2
        # Let X = num*c - den, Y = num*d - den.
        # XY = den^2.
        # c = (X + den) / num, d = (Y + den) / num.
        
        D_sq = den * den
        fac_den = factorize(den)
        fac_D_sq = tuple((p, 2*e) for p, e in fac_den)
        divs = get_divisors(fac_D_sq)
        
        valid_pairs = []
        
        lower_x = num * min_den - den
        upper_y = num * max_den_limit - den
        
        for x in divs:
            if x >= den:
                continue
            if x <= lower_x:
                continue
            
            y = D_sq // x
            
            if y > upper_y:
                continue
            
            if (x + den) % num != 0:
                continue
            c = (x + den) // num
            
            if c <= min_den:
                continue
                
            if (y + den) % num != 0:
                continue
            d = (y + den) // num
            
            if d <= c:
                continue
            if d > max_den_limit:
                continue
            
            valid_pairs.append((c, d))
            
        return valid_pairs

    # 1. Search 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    for a in range(7, 19):
        if time.time() > deadline - 5.0:
            break
        
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        max_b = int(2 / rem1)
        
        for b in range(a + 1, max_b + 1):
            if time.time() > deadline - 2.0:
                break
            
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            if rem2.denominator == 1:
                c = rem2.numerator
                if c > b and c < best_max:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                        best_max = c
                        best_solution = [a, b, c]
            else:
                if rem2.denominator % rem2.numerator == 0:
                    c = rem2.denominator // rem2.numerator
                    if c > b and c < best_max:
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                            best_max = c
                            best_solution = [a, b, c]

    # 2. Search 4-term solutions
    if best_solution is not None:
        limit = best_max - 1
        
        for a in range(7, 25):
            if time.time() > deadline - 5.0:
                break
            
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            max_b = int(3 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 3.0:
                    break
                
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                pairs = find_2_term_solutions(rem2, b, limit)
                if pairs:
                    min_d = float('inf')
                    best_c = None
                    for c, d in pairs:
                        if d < min_d:
                            min_d = d
                            best_c = c
                    
                    if min_d < best_max:
                        sol = [a, b, best_c, min_d]
                        if sum(Fraction(1, x) for x in sol) == target:
                            best_max = min_d
                            best_solution = sol

    # 3. Search 5-term solutions
    if best_solution is not None and best_max > 50 and time.time() < deadline - 10.0:
        limit_5 = best_max - 1
        
        for a in range(7, 32):
            if time.time() > deadline - 5.0:
                break
            
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            max_b = int(4 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 3.0:
                    break
                
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                max_c = int(3 / rem2)
                
                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline - 1.5:
                        break
                    
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    
                    pairs = find_2_term_solutions(rem3, c, limit_5)
                    if pairs:
                        min_e = float('inf')
                        best_d = None
                        for d, e in pairs:
                            if e < min_e:
                                min_e = e
                                best_d = d
                        
                        if min_e < best_max:
                            sol = [a, b, c, best_d, min_e]
                            if sum(Fraction(1, x) for x in sol) == target:
                                best_max = min_e
                                best_solution = sol

    # Fallback
    if best_solution is None:
        best_solution = [8, 28, 1736]
        best_max = 1736

    return best_solution
