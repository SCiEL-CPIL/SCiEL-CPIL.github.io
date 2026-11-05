def construct():
    import time
    from fractions import Fraction
    from math import isqrt, gcd

    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    target = Fraction(5, 31)

    best_max_den = 10000001
    best_solution = None

    # Strategy 1: Exhaustive 3-term search via factorization
    # For 1/a + 1/b + 1/c = 5/31 with a <= b <= c:
    # a is in [7, 18] (from 1/a < 5/31 and 3/a >= 5/31)
    # But let's be more generous: a from 7 to 200
    
    for a in range(7, 200):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
            
        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a
        
        g = gcd(num, den)
        n = num // g
        d = den // g
        
        d2 = d * d
        limit = isqrt(d2)
        
        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u
            
            bu = u + d
            bv = v + d
            
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                
                if b <= 0 or c <= 0:
                    continue
                if b == a or c == a or b == c:
                    continue
                if b > 10000000 or c > 10000000:
                    continue
                
                m = max(a, b, c)
                if m < best_max_den:
                    check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                    if check == target:
                        best_max_den = m
                        best_solution = [a, b, c]

    # Strategy 2: Bounded DFS with tight mathematical bounds
    # This can find 4+, 5+ term solutions with smaller max denominator
    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        nonlocal best_max_den, best_solution
        if time.time() - start_time > DEADLINE:
            return False
        if r == 0:
            if current:
                m = max(current)
                if m < best_max_den:
                    best_max_den = m
                    best_solution = list(current)
            return True
        if depth >= max_depth or r <= 0:
            return False
        
        # Tight bound: d must satisfy 1/d <= r, so d >= ceil(1/r)
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        # Additional pruning: if we have remaining terms, the smallest possible
        # sum of reciprocals from d_start upward must be at least r
        # This is a loose bound but helps
        
        for d in range(d_start, max_den + 1):
            if time.time() - start_time > DEADLINE:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                continue
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False

    # Try DFS with decreasing bounds starting from current best
    # If we found a 3-term solution, try to beat it with 4+ terms
    search_max = min(best_max_den - 1, 500)
    while search_max > 50:
        if time.time() - start_time > DEADLINE:
            break
        if dfs(target, 1, search_max, [], 0):
            break
        search_max -= 5

    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution

    # Fallback: greedy Egyptian fraction
    r = Fraction(5, 31)
    values = []
    used = set()
    while r > 0 and len(values) < 256:
        d = (r.denominator + r.numerator - 1) // r.numerator
        while d in used:
            d += 1
        values.append(d)
        used.add(d)
        r -= Fraction(1, d)
    return values
