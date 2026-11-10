import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    target = Fraction(5, 31)
    best_max_denom = 10000000
    best_denoms = []
    
    # Search for 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a <= 18 (since 3/a >= 5/31 => a <= 18.6)
    # a > 6 (since 1/a < 5/31 => a > 6.2)
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        r_ab = target - Fraction(1, a)
        if r_ab <= 0:
            continue
        # 1/b + 1/c = r_ab
        # b <= 2 / r_ab
        # b > 1 / r_ab
        # b >= a + 1
        
        b_min = a + 1
        b_max = int(2 / r_ab)
        
        if b_max < b_min:
            continue
            
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            r_c = r_ab - Fraction(1, b)
            if r_c <= 0:
                continue
            c = 1 / r_c
            if c.denominator == 1:
                c_val = c.numerator
                if c_val > b and c_val < 10000000:
                    if c_val < best_max_denom:
                        best_max_denom = c_val
                        best_denoms = [a, b, c_val]
    
    # Search for 4-term solutions with small denominators
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # Limit search space to keep it fast
    # a up to 20, b up to 50, c up to 100
    
    if time.time() < deadline - 10:
        for a in range(7, 21):
            if time.time() > deadline - 5:
                break
            r_bcd = target - Fraction(1, a)
            if r_bcd <= 0:
                continue
            for b in range(a + 1, 51):
                if time.time() > deadline - 5:
                    break
                r_cd = r_bcd - Fraction(1, b)
                if r_cd <= 0:
                    continue
                for c in range(b + 1, 101):
                    if time.time() > deadline - 5:
                        break
                    r_d = r_cd - Fraction(1, c)
                    if r_d <= 0:
                        continue
                    d = 1 / r_d
                    if d.denominator == 1:
                        d_val = d.numerator
                        if d_val > c and d_val < 10000000:
                            if d_val < best_max_denom:
                                best_max_denom = d_val
                                best_denoms = [a, b, c, d_val]
    
    if not best_denoms:
        # Fallback to greedy if no solution found (should not happen)
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
        
    return best_denoms
