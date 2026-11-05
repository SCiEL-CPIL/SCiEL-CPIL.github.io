<<<<<<< SEARCH
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
=======
    # Search 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a >= 7, a <= 18
    # For each a, rem1 = 5/31 - 1/a.
    # Find 1/b + 1/c = rem1 with b > a, c > b.
    # We want to minimize c.
    # Use divisor-based pair enumeration for efficiency.
    
    for a in range(7, 19):
        if time.time() > deadline - 5.0:
            break
        
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # Use find_2_term_solutions to find (b, c) with b > a, c < best_max
        limit = best_max - 1
        pairs = find_2_term_solutions(rem1, a, limit)
        if pairs:
            # Find pair with smallest c (max denominator)
            min_c_pair = min(pairs, key=lambda x: x[1])
            b, c = min_c_pair
            if is_valid([a, b, c]):
                curr_max = c
                if curr_max < best_max:
                    best_max = curr_max
                    best_solution = [a, b, c]
>>>>>>> REPLACE

