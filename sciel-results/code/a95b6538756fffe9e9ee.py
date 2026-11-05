def construct():
    import time
    from math import gcd, isqrt
    
    start_time = time.time()
    DEADLINE = 85.0
    
    target_num = 5
    target_den = 31
    
    best_max_den = 10**9
    best_solution = None
    
    def update_best(sol):
        nonlocal best_max_den, best_solution
        if not sol:
            return
        # Check distinctness
        if len(set(sol)) != len(sol):
            return
        # Check range
        if any(d < 1 or d > 10000000 for d in sol):
            return
        # Verify sum exactly
        # Sum 1/d_i = target_num/target_den
        # Sum target_den * prod(d_j != i) / prod(d_j) = target_num * prod(d_j) / prod(d_j)
        # Let P = prod(d_j). Then sum target_den * P/d_i = target_num * P
        # We can just use fractions for verification since N is small
        from fractions import Fraction
        s = Fraction(0, 1)
        for d in sol:
            s += Fraction(1, d)
        if s != Fraction(target_num, target_den):
            return
        
        m = max(sol)
        if m < best_max_den:
            best_max_den = m
            best_solution = sol[:]
    
    def solve_2_term(rem_num, rem_den, existing):
        """
        Solve 1/b + 1/c = rem_num/rem_den.
        Returns list of (b, c) pairs.
        """
        if rem_num <= 0:
            return []
        g = gcd(rem_num, rem_den)
        n = rem_num // g
        d = rem_den // g
        
        # 1/b + 1/c = n/d => d(b+c) = nbc => nbc - db - dc = 0
        # (nb - d)(nc - d) = d^2
        # Let u = nb - d, v = nc - d. u*v = d^2.
        # b = (u+d)/n, c = (v+d)/n.
        # u must be a divisor of d^2.
        # Also b, c must be integers, so u+d must be divisible by n.
        
        results = []
        d2 = d * d
        # Iterate divisors u of d^2
        # We only need to check u up to isqrt(d^2) = d, but u can be negative?
        # Since b,c > 0, nb > d => u > 0. So u is positive divisor.
        
        # Optimization: u must satisfy u ≡ -d (mod n).
        # So u = k*n - d for some integer k.
        # Also u*v = d^2.
        
        # Simple loop over divisors is fast enough for small d.
        for u in range(1, isqrt(d2) + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            
            # Check divisibility
            bu = u + d
            bv = v + d
            
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                
                if b <= 0 or c <= 0:
                    continue
                
                # Ensure distinctness with existing and each other
                # We will handle distinctness in the caller, but let's filter here if easy
                # Actually, caller passes 'existing' list.
                
                results.append((b, c))
        return results

    # Phase 1: 2-term search
    # 1/a + 1/b = 5/31
    sols_2 = solve_2_term(5, 31, [])
    for b, c in sols_2:
        update_best([b, c])
        
    # Phase 2: 3-term search
    # 1/a + 1/b + 1/c = 5/31
    # a must be such that 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # Also, to minimize max, we can assume a <= b <= c.
    # Then 3/a >= 5/31 => a <= 93/5 = 18.6 => a <= 18.
    # So a is in [7, 18].
    
    for a in range(7, 19):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
            
        # rem = 5/31 - 1/a = (5a - 31) / (31a)
        rem_num = 5 * a - 31
        rem_den = 31 * a
        
        if rem_num <= 0:
            continue
            
        sols_2 = solve_2_term(rem_num, rem_den, [a])
        for b, c in sols_2:
            if b == a or c == a or b == c:
                continue
            # Also ensure b, c are distinct from each other (handled in solve_2_term? No, solve_2 returns pairs, might be same if d^2 is square and u=v)
            # If u=v, then b=c. We check b!=c here.
            update_best([a, b, c])
            
    # Phase 3: 4-term search
    # If best_max_den is still large, try 4 terms.
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Assume a <= b <= c <= d.
    # 4/a >= 5/31 => a <= 124/5 = 24.8 => a <= 24.
    # a >= 7.
    
    if best_max_den > 20:
        for a in range(7, 25):
            if time.time() - start_time > DEADLINE:
                break
            if a >= best_max_den:
                break
                
            rem1_num = 5 * a - 31
            rem1_den = 31 * a
            if rem1_num <= 0:
                continue
                
            # Now solve 1/b + 1/c + 1/d = rem1
            # b must be such that 1/b < rem1 => b > rem1_den/rem1_num
            # Also 3/b >= rem1 => b <= 3*rem1_den/rem1_num
            # Let's just iterate b from a+1 upwards.
            # To keep it bounded, if b is too large, max will be large.
            # We stop if b >= best_max_den.
            
            for b in range(a + 1, best_max_den):
                if time.time() - start_time > DEADLINE:
                    break
                if b >= best_max_den:
                    break
                
                # rem2 = rem1 - 1/b = (rem1_num*b - rem1_den) / (rem1_den*b)
                rem2_num = rem1_num * b - rem1_den
                rem2_den = rem1_den * b
                
                if rem2_num <= 0:
                    break # b is too large, rem becomes negative or zero, so larger b will also fail
                
                # Solve 1/c + 1/d = rem2
                sols_2 = solve_2_term(rem2_num, rem2_den, [a, b])
                for c, d in sols_2:
                    if c == a or c == b or d == a or d == b or c == d:
                        continue
                    if c < b: # We assumed a<=b<=c<=d, so c should be >= b? Not necessarily, but for uniqueness we can enforce c>=b
                         # Actually, solve_2_term finds all pairs. We just need distinct values.
                         # If c < b, it's the same set {a,b,c,d}. We can just check distinctness.
                         pass
                    update_best([a, b, c, d])
                    
    if best_solution is not None:
        return best_solution
        
    # Fallback: Greedy (should not be reached if logic is correct for small numbers)
    from fractions import Fraction
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
        if len(values) > 10: break
    return values
