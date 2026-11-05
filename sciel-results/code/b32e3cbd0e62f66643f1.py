def construct():
    import time
    from fractions import Fraction
    from math import isqrt

    start_time = time.time()
    # The evaluator kills at 90s. We aim to finish well before to be safe.
    # 85s is a good hard limit, 80s for search.
    deadline = start_time + 85.0

    target_num = 5
    target_den = 31
    target = Fraction(target_num, target_den)
    
    # Known good solution: 5/31 = 1/8 + 1/28 + 1/1736
    # Max is 1736.
    best_solution = [8, 28, 1736]
    best_max = 1736

    def is_valid(denoms):
        if not denoms or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 1 or d > 10000000 for d in denoms):
            return False
        # Check sum exactly
        s = sum(Fraction(1, d) for d in denoms)
        return s == target

    def factorize(n):
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
        Find pairs (c, d) with min_den < c < d <= max_den_limit
        such that 1/c + 1/d = rem.
        Returns list of (c, d) or None if none found.
        Uses divisor generation of D^2.
        """
        num, den = rem.numerator, rem.denominator
        # (num*c - den)(num*d - den) = den^2
        # Let X = num*c - den, Y = num*d - den.
        # XY = den^2.
        # c = (X + den) / num, d = (Y + den) / num.
        
        D_sq = den * den
        fac_den = factorize(den)
        fac_D_sq = {p: 2*e for p, e in fac_den.items()}
        divs = get_divisors(fac_D_sq)
        
        valid_pairs = []
        
        # We want X < Y = D_sq/X. So X < sqrt(D_sq) = den.
        # Also c > min_den => X > num*min_den - den.
        # And d <= max_den_limit => Y <= num*max_den_limit - den.
        
        lower_x = num * min_den - den
        upper_y = num * max_den_limit - den
        
        # Sort divisors to allow early termination if sorted ascending? 
        # Actually, we just iterate. The number of divisors is small.
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

    # Search 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a >= 7, a <= 18
    # For each a, rem1 = 5/31 - 1/a.
    # Find 1/b + 1/c = rem1 with b > a, c > b.
    # We want to minimize c.
    
    for a in range(7, 19):
        if time.time() > deadline - 5.0:
            break
        
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # b > a
        # 2/b >= rem1 => b <= 2/rem1
        max_b = int(2 / rem1)
        
        for b in range(a + 1, max_b + 1):
            if time.time() > deadline - 2.0:
                break
            
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            # 1/c = rem2
            # c must be integer.
            # rem2 = N/D. We need N/D = 1/k => D = N*k.
            # So D must be divisible by N.
            if rem2.denominator % rem2.numerator == 0:
                c = rem2.denominator // rem2.numerator
                if c > b and c < best_max:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                        if c < best_max:
                            best_max = c
                            best_solution = [a, b, c]

    # Search 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # We want to find any solution with max < best_max.
    # a < b < c < d < best_max.
    
    if best_solution is not None:
        limit = best_max - 1
        
        # a range
        # 4/a >= 5/31 => a <= 124/5 = 24.8 => a <= 24
        # But 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
        for a in range(7, 25):
            if time.time() > deadline - 5.0:
                break
            
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # 1/b + 1/c + 1/d = rem1
            # b > a
            # 3/b >= rem1 => b <= 3/rem1
            max_b = int(3 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 3.0:
                    break
                
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # 1/c + 1/d = rem2
                # c > b
                # d < limit
                # We use find_2_term_solutions(rem2, b, limit)
                
                pairs = find_2_term_solutions(rem2, b, limit)
                if pairs:
                    # We want the pair with smallest d? 
                    # The function returns all valid pairs.
                    # We want to minimize the max denominator, which is d.
                    # So we pick the pair with min d.
                    min_d_pair = min(pairs, key=lambda x: x[1])
                    c, d = min_d_pair
                    if is_valid([a, b, c, d]):
                        curr_max = d
                        if curr_max < best_max:
                            best_max = curr_max
                            best_solution = [a, b, c, d]

    # Search 5-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # We want to find any solution with max < best_max.
    # a < b < c < d < e < best_max.
    
    if best_solution is not None:
        limit = best_max - 1
        
        # a range
        # 5/a >= 5/31 => a <= 31
        # But 1/a < 5/31 => a > 6.2 => a >= 7
        for a in range(7, 32):
            if time.time() > deadline - 5.0:
                break
            
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # 1/b + 1/c + 1/d + 1/e = rem1
            # b > a
            # 4/b >= rem1 => b <= 4/rem1
            max_b = int(4 / rem1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline - 3.0:
                    break
                
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # 1/c + 1/d + 1/e = rem2
                # c > b
                # 3/c >= rem2 => c <= 3/rem2
                max_c = int(3 / rem2)
                
                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline - 2.0:
                        break
                    
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    
                    # 1/d + 1/e = rem3
                    # d > c
                    # e < limit
                    # We use find_2_term_solutions(rem3, c, limit)
                    
                    pairs = find_2_term_solutions(rem3, c, limit)
                    if pairs:
                        min_e_pair = min(pairs, key=lambda x: x[1])
                        d, e = min_e_pair
                        if is_valid([a, b, c, d, e]):
                            curr_max = e
                            if curr_max < best_max:
                                best_max = curr_max
                                best_solution = [a, b, c, d, e]

    return best_solution
