import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80.0
    target = Fraction(5, 31)
    
    # Known fallback
    best_sol = [8, 31, 248]
    best_max = 248
    
    # Systematic 3-term search
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    # a must satisfy 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # Also 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18 (since 3 terms, each <= 1/a)
    # Wait, that's not right. We need 1/a + 1/b + 1/c = 5/31 with a<b<c.
    # The maximum sum with a as smallest is 1/a + 1/(a+1) + 1/(a+2). Need this >= 5/31.
    # The minimum sum with a as smallest (for fixed a) is 1/a + 1/(a+1) + 1/(a+2) approximately.
    # Actually for fixed a, b and c can be large, so sum can be close to 1/a.
    # Need 1/a < 5/31 (since other terms are positive), so a > 31/5 = 6.2, a >= 7.
    # Need 1/a + 1/(a+1) + 1/(a+2) >= 5/31 for feasibility.
    
    for a in range(7, 100):  # a is bounded above
        if time.time() > deadline:
            break
        
        # Check if 3 terms starting with a can reach 5/31
        # Max possible: 1/a + 1/(a+1) + 1/(a+2)
        max_sum = Fraction(1, a) + Fraction(1, a+1) + Fraction(1, a+2)
        if max_sum < target:
            continue
        
        # Min possible with a: 1/a + 1/M + 1/M for large M, approaches 1/a
        # Need 1/a < 5/31, already ensured by a >= 7
        # For fixed a, 1/b + 1/c = 5/31 - 1/a = rem
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        
        # 1/b + 1/c = rem, b > a, c > b
        # (b+c)/(bc) = rem => rem*bc = b+c => rem*bc - b - c = 0
        # b(rem*c - 1) = c => b = c/(rem*c - 1)
        # For b to be integer > a: rem*c - 1 must divide c, and result > a
        # Also c > b > a
        
        # b = c / (rem*c - 1) = c / ((rem.numerator*c - rem.denominator)/rem.denominator)
        # = c * rem.denominator / (rem.numerator*c - rem.denominator)
        # Let num = rem.numerator, den = rem.denominator
        # b = c * den / (num*c - den)
        # Need num*c - den > 0 => c > den/num
        # And num*c - den | c*den
        
        num = rem.numerator
        den = rem.denominator
        
        # c > den/num, and c > a+1 (since c > b > a)
        c_min = max(a + 2, den // num + 1)
        
        # Upper bound on c: b > a means c*den/(num*c-den) > a
        # c*den > a*num*c - a*den
        # c*den + a*den > a*num*c
        # den*(c+a) > a*num*c
        # c*den + a*den > a*num*c
        # a*den > c*(a*num - den)
        # If a*num - den > 0: c < a*den/(a*num - den)
        # If a*num - den <= 0: no upper bound from this
        
        a_num = a * num
        if a_num > den:
            c_max = (a * den) // (a_num - den) + 1
        else:
            c_max = 2000000  # large fallback
        
        for c in range(c_min, min(c_max + 1, 10000000)):
            if time.time() > deadline:
                break
            denom_b = num * c - den
            if denom_b <= 0:
                continue
            # b = c * den / denom_b
            if (c * den) % denom_b != 0:
                continue
            b = (c * den) // denom_b
            if b <= a:
                continue
            if b >= c:
                continue
            # Verify
            sol = [a, b, c]
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s == target:
                m = c
                if m < best_max:
                    best_max = m
                    best_sol = sol
    
    # Verify final solution
    s = sum(Fraction(1, x) for x in best_sol)
    if s != target or len(best_sol) < 1 or len(best_sol) > 256:
        return [8, 31, 248]
    if len(set(best_sol)) != len(best_sol):
        return [8, 31, 248]
    if any(d < 1 or d > 10000000 for d in best_sol):
        return [8, 31, 248]
    
    return best_sol
