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
        # Verify sum using integer arithmetic
        # Sum of 1/d for d in denoms == 5/31
        # Use Fraction for exact check
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
        # 1/c + 1/d = p/q => q(c+d) = p*c*d => p*c*d - q*c - q*d = 0
        # (p*c - q)(p*d - q) = q^2
        # Let X = p*c - q, Y = p*d - q, then X*Y = q^2
        # c = (X + q) / p, d = (Y + q) / p
        # Need X + q > 0, Y + q > 0, both divisible by p
        # X < Y (since c < d), X > 0 (since c >= 1 means X = p*c - q >= p - q)
        
        q2 = q * q
        # Enumerate divisors X of q^2 where X < q (since c < d implies X < Y, and X*Y=q^2, so X < q)
        # Actually X*Y = q^2 and X < Y means X < q (since if X >= q then Y = q^2/X <= q, contradiction)
        
        # Find all divisors of q^2
        # q can be large, so we need to be careful
        # q = 31*a for 3-term case, or smaller for 4-term
        # Let's iterate through possible X values
        
        # X must divide q^2, X < q, and (X + q) % p == 0
        # Also c = (X+q)/p >= min_d, d = (Y+q)/p <= max_d, d > c
        
        # Efficient: iterate X from 1 to sqrt(q^2) = q, check if X divides q^2
        # But q can be up to 31*50 = 1550 for 3-term, so q^2 ~ 2.4M, sqrt ~ 1550
        # For 4-term, q is smaller
        
        # Actually, let's just iterate through divisors properly
        # For efficiency, factor q and generate divisors of q^2
        
        # Simpler approach for reasonable q: iterate X from 1 to q-1, check divisibility
        # This is O(q) which is fine for q < 10000
        
        if q > 100000:
            # Too large, use a different approach
            # For now, skip
            return []
        
        for X in range(1, q):
            if q2 % X != 0:
                continue
            Y = q2 // X
            if Y <= X:
                continue
            # Check divisibility
            c_num = X + q
            d_num = Y + q
            if c_num % p != 0 or d_num % p != 0:
                continue
            c = c_num // p
            d = d_num // p
            if c < min_d or d > max_d or d <= c:
                continue
            if c >= 1 and d >= 1:
                solutions.append((c, d))
        
        return solutions
    
    # Strategy 1: Exhaustive 3-term search
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    # a >= ceil(31/5) = 7
    # For each a, rem = 5/31 - 1/a = (5a-31)/(31a)
    # Need 1/b + 1/c = rem
    # a can go up to... well, we want to minimize max(c), so try small a first
    
    max_a = 200  # Reasonable upper bound for a
    
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
                if best_max <= 100:  # Very good, stop early
                    break
    
    # Strategy 2: 4-term search
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Pick a and b, then solve 1/c + 1/d = rem
    # a >= 7, b > a
    # For efficiency, limit a and b ranges
    
    if time.time() < deadline - 10:
        max_a4 = 30
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
            # b upper bound: we need at least 2 more terms, each >= 1/(max_d)
            # rem_a - 1/b >= 2/max_d... but let's just limit b reasonably
            # b can't be too large because then rem_a - 1/b is close to rem_a
            # and we need 2 terms to sum to that
            b_hi = min(200, 10000)
            
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
        # Pick a, b, c, then solve 1/d + 1/e = rem
        max_a5 = 15
        for a in range(7, max_a5 + 1):
            if time.time() > deadline:
                break
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            b_lo = a + 1
            b_hi = min(50, 1000)
            for b in range(b_lo, b_hi + 1):
                if time.time() > deadline:
                    break
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    break
                c_lo = b + 1
                c_hi = min(100, 1000)
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
    
    # Fallback
    if best_solution is None:
        best_solution = [7, 56, 1736]
        best_max = 1736
    
    return best_solution
