def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    from fractions import Fraction
    from math import isqrt, gcd

    target = Fraction(5, 31)
    best_max_den = 10000001
    best_solution = None

    def check_solution(denoms):
        nonlocal best_max_den, best_solution
        if len(denoms) < 1 or len(denoms) > 256:
            return
        if len(set(denoms)) != len(denoms):
            return
        for d in denoms:
            if d < 1 or d > 10000000:
                return
        s = sum(Fraction(1, d) for d in denoms)
        if s == target:
            m = max(denoms)
            if m < best_max_den:
                best_max_den = m
                best_solution = list(denoms)

    def solve_2term(r):
        """Solve 1/b + 1/c = r where r = n/d reduced. Return (b,c) or None."""
        if r <= 0:
            return None
        n = r.numerator
        d = r.denominator
        d2 = d * d
        limit = isqrt(d2)
        results = []
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            bu = u + d
            bv = v + d
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                if b > 0 and c > 0:
                    results.append((b, c))
        return results

    def greedy_complete(r, min_den, used, max_terms=4):
        """Complete r using greedy method, denominators >= min_den, distinct from used.
        Returns list of denominators or None."""
        if r <= 0:
            return [] if r == 0 else None
        if max_terms <= 0:
            return None
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d < min_den:
            d = min_den
        if d in used:
            # Try next few values
            found = None
            for dd in range(d, d + 50):
                if dd in used:
                    continue
                new_r = r - Fraction(1, dd)
                if new_r < 0:
                    break
                if new_r == 0:
                    return [dd]
                rest = greedy_complete(new_r, dd + 1, used | {dd}, max_terms - 1)
                if rest is not None:
                    return [dd] + rest
            return None
        new_r = r - Fraction(1, d)
        if new_r < 0:
            return None
        if new_r == 0:
            return [d]
        rest = greedy_complete(new_r, d + 1, used | {d}, max_terms - 1)
        if rest is not None:
            return [d] + rest
        return None

    # Phase 1: Systematic search over pairs (a, b) with a <= b
    # For each pair, check if remaining can be completed with 1, 2, or 3 more terms
    # WLOG a <= b <= c <= d for 4 terms, or a <= b <= c for 3 terms
    
    # a must satisfy: 1/a < 5/31 => a >= 7
    # For 3 terms: a <= b <= c, so 3/a >= 5/31 => a <= 93/5 = 18.6, a <= 18
    # For 4 terms: a <= b <= c <= d, so 4/a >= 5/31 => a <= 124/5 = 24.8, a <= 24
    # But we should search more broadly to find optimal
    
    # Search a from 7 up to a reasonable limit
    # For minimizing max, we want all denominators small
    # Let's search a up to 100, b up to max(a, best_max_den-1)
    
    for a in range(7, 101):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        
        # b >= a, and 1/b < 5/31 - 1/a
        # Also for 3 terms: 2/b >= 5/31 - 1/a => b <= 2/(5/31 - 1/a)
        # For 4 terms: 3/b >= 5/31 - 1/a => b <= 3/(5/31 - 1/a)
        
        r_after_a = target - Fraction(1, a)
        if r_after_a <= 0:
            continue
        
        # b must satisfy: 1/b < r_after_a => b > 1/r_after_a
        # And for remaining terms to be feasible:
        # If we use 2 more terms (3 total): 2/b >= r_after_a => b <= 2/r_after_a
        # If we use 3 more terms (4 total): 3/b >= r_after_a => b <= 3/r_after_a
        
        b_lo = max(a, r_after_a.denominator // r_after_a.numerator + 1)
        b_hi_3 = (2 * r_after_a.denominator) // r_after_a.numerator  # for 3-term total
        b_hi_4 = (3 * r_after_a.denominator) // r_after_a.numerator  # for 4-term total
        b_hi = min(best_max_den - 1, b_hi_4)
        
        if b_lo > b_hi:
            continue
        
        for b in range(b_lo, b_hi + 1):
            if time.time() - start_time > DEADLINE:
                break
            if b >= best_max_den:
                break
            if b == a:
                continue
            
            r_after_ab = r_after_a - Fraction(1, b)
            if r_after_ab < 0:
                break  # b too small, but since b is increasing, r_after_ab decreasing... 
                        # actually r_after_ab = r_after_a - 1/b, as b increases, 1/b decreases,
                        # so r_after_ab increases. So we shouldn't break here.
                        # Wait, let me reconsider.
            # r_after_ab = 5/31 - 1/a - 1/b
            # As b increases, 1/b decreases, so r_after_ab increases.
            # So if r_after_ab < 0, it will stay < 0 for larger b? No!
            # r_after_ab increases with b, so if it's < 0 now, it might become >= 0 later.
            # Actually no: r_after_ab = constant - 1/b. As b increases, 1/b decreases,
            # so r_after_ab increases. So if r_after_ab < 0, it will eventually become >= 0.
            # We should NOT break.
            # But we can skip this b.
            pass
            if r_after_ab == 0:
                # 2-term solution: a, b
                check_solution([a, b])
                continue
            
            if r_after_ab > 0:
                # Try to complete with 1 more term (3-term total)
                # 1/c = r_after_ab => c = 1/r_after_ab
                c_val = r_after_ab.denominator // r_after_ab.numerator
                if r_after_ab.numerator == 1:
                    c = r_after_ab.denominator
                    if c > b and c != a and c <= 10000000:
                        check_solution([a, b, c])
                
                # Try to complete with 2 more terms (4-term total)
                # Solve 1/c + 1/d = r_after_ab
                pairs = solve_2term(r_after_ab)
                if pairs:
                    for (c, d) in pairs:
                        if c > b and d > b and c != a and d != a and c != d:
                            if c <= 10000000 and d <= 10000000:
                                check_solution([a, b, c, d])
                
                # Try greedy completion for up to 3 more terms
                if time.time() - start_time < DEADLINE - 5:
                    used = {a, b}
                    rest = greedy_complete(r_after_ab, b + 1, used, max_terms=3)
                    if rest is not None:
                        check_solution([a, b] + rest)

    if best_solution is not None:
        return best_solution

    # Phase 2: Broader search with larger a values
    for a in range(101, 500):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        
        r_after_a = target - Fraction(1, a)
        if r_after_a <= 0:
            continue
        
        b_lo = max(a, r_after_a.denominator // r_after_a.numerator + 1)
        b_hi_4 = (3 * r_after_a.denominator) // r_after_a.numerator
        b_hi = min(best_max_den - 1, b_hi_4)
        
        if b_lo > b_hi:
            continue
        
        for b in range(b_lo, b_hi + 1):
            if time.time() - start_time > DEADLINE:
                break
            if b >= best_max_den:
                break
            if b == a:
                continue
            
            r_after_ab = r_after_a - Fraction(1, b)
            if r_after_ab <= 0:
                continue
            
            # Try 2-term completion
            pairs = solve_2term(r_after_ab)
            if pairs:
                for (c, d) in pairs:
                    if c > b and d > b and c != a and d != a and c != d:
                        if c <= 10000000 and d <= 10000000:
                            check_solution([a, b, c, d])
            
            # Try 1-term completion
            if r_after_ab.numerator == 1:
                c = r_after_ab.denominator
                if c > b and c != a and c <= 10000000:
                    check_solution([a, b, c])

    if best_solution is not None:
        return best_solution

    # Fallback: greedy
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
