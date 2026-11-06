import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    target = Fraction(5, 31)
    target_num = 5
    target_den = 31

    best_solution = None
    best_max = float('inf')

    def update_best(denoms):
        nonlocal best_solution, best_max
        if not denoms or len(denoms) > 256:
            return
        if len(set(denoms)) != len(denoms):
            return
        if any(d < 1 or d > 10000000 for d in denoms):
            return
        # Verify sum using integer arithmetic via Fraction
        s = sum(Fraction(1, d) for d in denoms)
        if s != target:
            return
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_solution = list(denoms)

    def find_2term_solutions(rem, min_d, max_d):
        """Find all pairs (c, d) with c < d, c >= min_d, d <= max_d such that 1/c + 1/d = rem."""
        if rem <= 0:
            return []
        solutions = []
        p = rem.numerator
        q = rem.denominator
        if p <= 0:
            return []
        
        q2 = q * q
        # Enumerate divisors X of q^2 where X < q
        # c = (X + q) / p, d = (Y + q) / p where Y = q^2 / X
        # Need X + q divisible by p, Y + q divisible by p
        # c >= min_d, d <= max_d, c < d
        
        if q > 200000:
            return []
        
        for X in range(1, q):
            if q2 % X != 0:
                continue
            Y = q2 // X
            if Y <= X:
                continue
            c_num = X + q
            d_num = Y + q
            if c_num % p != 0 or d_num % p != 0:
                continue
            c = c_num // p
            d = d_num // p
            if c < min_d or d > max_d or d <= c:
                continue
            solutions.append((c, d))
        
        return solutions

    # Strategy 1: Exhaustive 3-term search
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    max_a = 300

    for a in range(7, max_a + 1):
        if time.time() > deadline:
            break
        # rem = 5/31 - 1/a = (5a - 31) / (31a)
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
        # Simplify fraction
        g = gcd(p, q)
        p_s = p // g
        q_s = q // g
        
        # Find 2-term solutions for rem = p_s/q_s with b > a, c > b
        sols = find_2term_solutions(Fraction(p_s, q_s), a + 1, 10000000)
        for (b, c) in sols:
            if time.time() > deadline:
                break
            denoms = [a, b, c]
            mx = max(denoms)
            if mx < best_max:
                update_best(denoms)
                if best_max <= 50:
                    break
    
    # Strategy 2: 4-term search
    # 1/a + 1/b + 1/c + 1/d = 5/31
    if time.time() < deadline - 15:
        max_a4 = 40
        for a in range(7, max_a4 + 1):
            if time.time() > deadline:
                break
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            # b > a, 1/b < rem_a => b > 1/rem_a
            b_min = a + 1
            # b must satisfy 1/b <= rem_a, so b >= ceil(1/rem_a)
            b_lo = max(b_min, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator)
            # b upper bound
            b_hi = min(300, 10000)
            
            for b in range(b_lo, b_hi + 1):
                if time.time() > deadline:
                    break
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    break
                # Find 1/c + 1/d = rem_b with c > b, d > c
                sols = find_2term_solutions(rem_b, b + 1, 10000000)
                for (c, d) in sols:
                    if time.time() > deadline:
                        break
                    denoms = [a, b, c, d]
                    mx = max(denoms)
                    if mx < best_max:
                        update_best(denoms)
    
    # Strategy 3: 5-term search if needed
    if best_max > 100 and time.time() < deadline - 5:
        max_a5 = 20
        for a in range(7, max_a5 + 1):
            if time.time() > deadline:
                break
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            b_lo = a + 1
            b_hi = min(60, 1000)
            for b in range(b_lo, b_hi + 1):
                if time.time() > deadline:
                    break
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    break
                c_lo = b + 1
                c_hi = min(150, 1000)
                for c in range(c_lo, c_hi + 1):
                    if time.time() > deadline:
                        break
                    rem_c = rem_b - Fraction(1, c)
                    if rem_c <= 0:
                        break
                    sols = find_2term_solutions(rem_c, c + 1, 10000000)
                    for (d, e) in sols:
                        if time.time() > deadline:
                            break
                        denoms = [a, b, c, d, e]
                        mx = max(denoms)
                        if mx < best_max:
                            update_best(denoms)
    
    # Fallback: Greedy completion
    if best_solution is None or best_max > 1000:
        # Try greedy from various starting points
        for first_d in range(7, 100):
            if time.time() > deadline - 1:
                break
            rem = target - Fraction(1, first_d)
            if rem < 0:
                break
            used = {first_d}
            vals = [first_d]
            d_min = first_d + 1
            while rem > 0 and len(vals) < 256:
                d = (rem.denominator + rem.numerator - 1) // rem.numerator
                if d < d_min:
                    d = d_min
                if d in used:
                    d += 1
                    while d in used:
                        d += 1
                if d > 10000000:
                    break
                vals.append(d)
                used.add(d)
                rem -= Fraction(1, d)
                d_min = d + 1
            if rem == 0 and len(vals) <= 256:
                mx = max(vals)
                if mx < best_max:
                    update_best(vals)
    
    # Final fallback
    if best_solution is None:
        best_solution = [7, 56, 1736]
        best_max = 1736
    
    return best_solution
