def construct():
    import time
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_max = 248
    best_solution = [8, 31, 248]
    
    def try_update(a, b, c, d, e):
        nonlocal best_max, best_solution
        terms = [x for x in [a, b, c, d, e] if x is not None]
        if len(terms) < 2:
            return
        # Check distinctness
        if len(set(terms)) != len(terms):
            return
        if any(t < 1 or t > 10000000 for t in terms):
            return
        s = sum(Fraction(1, t) for t in terms)
        if s == target:
            m = max(terms)
            if m < best_max:
                best_max = m
                best_solution = list(terms)
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    # a must satisfy: 3/a >= 5/31 => a <= 93/5 = 18.6, so a <= 18
    # a must satisfy: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    for a in range(7, 19):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # 1/b + 1/c = r1, b > a, c > b
        # b must satisfy: 2/b >= r1 => b <= 2/r1
        # b must satisfy: 1/b < r1 => b > 1/r1
        # Also b > a
        b_lower = max(a + 1, int(1 / r1) + 1)
        b_upper = int(2 / r1) + 1
        if b_upper < b_lower:
            continue
        for b in range(b_lower, b_upper + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            # c = 1/r2 must be integer and c > b
            n2, d2 = r2.numerator, r2.denominator
            if n2 == 0:
                continue
            if d2 % n2 != 0:
                continue
            c = d2 // n2
            if c <= b:
                continue
            try_update(a, b, c, None, None)
    
    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a: 4/a >= 5/31 => a <= 124/5 = 24.8, so a <= 24
    # a: 1/a < 5/31 => a >= 7
    for a in range(7, 25):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # 1/b + 1/c + 1/d = r1, b > a
        # b: 3/b >= r1 => b <= 3/r1
        # b: 1/b < r1 => b > 1/r1
        b_lower = max(a + 1, int(1 / r1) + 1)
        b_upper = int(3 / r1) + 1
        if b_upper < b_lower:
            continue
        for b in range(b_lower, b_upper + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            # 1/c + 1/d = r2, c > b
            # c: 2/c >= r2 => c <= 2/r2
            # c: 1/c < r2 => c > 1/r2
            c_lower = max(b + 1, int(1 / r2) + 1)
            c_upper = int(2 / r2) + 1
            if c_upper < c_lower:
                continue
            for c in range(c_lower, c_upper + 1):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                n3, d3 = r3.numerator, r3.denominator
                if n3 == 0:
                    continue
                if d3 % n3 != 0:
                    continue
                d = d3 // n3
                if d <= c:
                    continue
                try_update(a, b, c, d, None)
    
    # Search for 5-term solutions
    # a: 5/a >= 5/31 => a <= 31
    # a: 1/a < 5/31 => a >= 7
    for a in range(7, 32):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # 1/b + 1/c + 1/d + 1/e = r1, b > a
        # b: 4/b >= r1 => b <= 4/r1
        b_lower = max(a + 1, int(1 / r1) + 1)
        b_upper = int(4 / r1) + 1
        if b_upper < b_lower:
            continue
        for b in range(b_lower, b_upper + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            # 1/c + 1/d + 1/e = r2, c > b
            # c: 3/c >= r2 => c <= 3/r2
            c_lower = max(b + 1, int(1 / r2) + 1)
            c_upper = int(3 / r2) + 1
            if c_upper < c_lower:
                continue
            for c in range(c_lower, c_upper + 1):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                # 1/d + 1/e = r3, d > c
                # d: 2/d >= r3 => d <= 2/r3
                d_lower = max(c + 1, int(1 / r3) + 1)
                d_upper = int(2 / r3) + 1
                if d_upper < d_lower:
                    continue
                for d in range(d_lower, d_upper + 1):
                    if time.time() > deadline:
                        break
                    r4 = r3 - Fraction(1, d)
                    if r4 <= 0:
                        continue
                    n4, d4 = r4.numerator, r4.denominator
                    if n4 == 0:
                        continue
                    if d4 % n4 != 0:
                        continue
                    e = d4 // n4
                    if e <= d:
                        continue
                    try_update(a, b, c, d, e)
    
    return best_solution
