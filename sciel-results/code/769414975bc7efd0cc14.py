def construct():
    import time
    from fractions import Fraction
    from math import isqrt

    start_time = time.time()
    deadline = start_time + 85.0

    target = Fraction(5, 31)
    best_solution = None
    best_max = float('inf')

    def factorize(n):
        if n <= 1:
            return {}
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, exp in factors.items():
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
        Find all pairs (c, d) with min_den < c < d <= max_den_limit
        such that 1/c + 1/d = rem.
        Uses (num*c - den)(num*d - den) = den^2 factorization.
        """
        num, den = rem.numerator, rem.denominator
        if num == 0:
            return []
        
        D_sq = den * den
        fac_den = factorize(den)
        fac_D_sq = {p: 2 * e for p, e in fac_den.items()}
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
            if d <= c or d > max_den_limit:
                continue
            valid_pairs.append((c, d))
        
        return valid_pairs

    # Step 1: Exhaustively find all 3-term solutions
    # 1/a + 1/b + 1/c = 5/31 with a < b < c
    # a must satisfy: 3/c <= 5/31 < 3/a => a > 93/5 = 18.6? No.
    # 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    # For each a, rem1 = 5/31 - 1/a, need 1/b + 1/c = rem1 with b > a
    
    for a in range(7, 19):
        if time.time() > deadline - 10.0:
            break
        
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # b > a, and 2/b >= rem1 => b <= 2/rem1
        max_b = int(2 / rem1)
        
        for b in range(a + 1, max_b + 1):
            if time.time() > deadline - 5.0:
                break
            
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            # Need 1/c = rem2, so c = 1/rem2 must be integer and c > b
            if rem2.denominator == 1:
                c = rem2.numerator
                if c > b:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                        if c < best_max:
                            best_max = c
                            best_solution = [a, b, c]
            else:
                if rem2.denominator % rem2.numerator == 0:
                    c = rem2.denominator // rem2.numerator
                    if c > b:
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                            if c < best_max:
                                best_max = c
                                best_solution = [a, b, c]

    # Step 2: Search for 4-term solutions with max denominator < best_max
    if best_solution is not None and best_max > 100:
        limit = best_max - 1
        
        # a: 4/a > 5/31 => a < 124/5 = 24.8, a >= 7
        for a in range(7, 25):
            if time.time() > deadline - 10.0:
                break
            
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # b: 3/b > rem1 => b < 3/rem1
            max_b = int(3 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 5.0:
                    break
                
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # Find 1/c + 1/d = rem2 with b < c < d < limit
                pairs = find_2_term_solutions(rem2, b, limit)
                if pairs:
                    # Find pair with smallest d
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

    # Step 3: Search for 5-term solutions with max denominator < best_max
    if best_solution is not None and best_max > 50 and time.time() < deadline - 15.0:
        limit = best_max - 1
        
        # a: 5/a > 5/31 => a < 31, a >= 7
        for a in range(7, 32):
            if time.time() > deadline - 10.0:
                break
            
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            max_b = int(4 / rem1)
            
            for b in range(a + 1, min(max_b + 1, 40)):
                if time.time() > deadline - 5.0:
                    break
                
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                max_c = int(3 / rem2)
                
                for c in range(b + 1, min(max_c + 1, 60)):
                    if time.time() > deadline - 3.0:
                        break
                    
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    
                    pairs = find_2_term_solutions(rem3, c, limit)
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

    # Fallback if no solution found (shouldn't happen)
    if best_solution is None:
        best_solution = [8, 31, 248]
        best_max = 248

    return best_solution
