"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Helper to check if a fraction is a unit fraction and return the denominator
    def get_denom_if_unit(frac):
        if frac.numerator == 1:
            return frac.denominator
        return None

    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # a must satisfy: 1/a < 5/31 => a > 6.2 => a >= 7
    # a must satisfy: 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b must satisfy: 2/b > rem_a => b < 2/rem_a
        # b must satisfy: 1/b < rem_a => b > 1/rem_a
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(2 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c = get_denom_if_unit(rem_b)
            if c is not None and c > b and c < best_max:
                best_max = c
                best_solution = [a, b, c]

    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a: 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
    # a: 1/a < 5/31 => a >= 7
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(3 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator + 1)
            c_max = min(best_max - 1, int(2 * rem_b.denominator / rem_b.numerator))
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                d = get_denom_if_unit(rem_c)
                if d is not None and d > c and d < best_max:
                    best_max = d
                    best_solution = [a, b, c, d]

    # Try 5-term solutions
    # a: 5/a > 5/31 => a < 31 => a <= 30
    for a in range(7, 31):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(4 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator + 1)
            c_max = min(best_max - 1, int(3 * rem_b.denominator / rem_b.numerator))
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                d_min = max(c + 1, (rem_c.denominator + rem_c.numerator - 1) // rem_c.numerator + 1)
                d_max = min(best_max - 1, int(2 * rem_c.denominator / rem_c.numerator))
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        break
                    e = get_denom_if_unit(rem_d)
                    if e is not None and e > d and e < best_max:
                        best_max = e
                        best_solution = [a, b, c, d, e]

    # Try 6-term solutions
    # a: 6/a > 5/31 => a < 186/5 = 37.2 => a <= 37
    for a in range(7, 38):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(5 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator + 1)
            c_max = min(best_max - 1, int(4 * rem_b.denominator / rem_b.numerator))
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                d_min = max(c + 1, (rem_c.denominator + rem_c.numerator - 1) // rem_c.numerator + 1)
                d_max = min(best_max - 1, int(3 * rem_c.denominator / rem_c.numerator))
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        break
                    e_min = max(d + 1, (rem_d.denominator + rem_d.numerator - 1) // rem_d.numerator + 1)
                    e_max = min(best_max - 1, int(2 * rem_d.denominator / rem_d.numerator))
                    
                    for e in range(e_min, e_max + 1):
                        if time.time() > deadline:
                            break
                        rem_e = rem_d - Fraction(1, e)
                        if rem_e <= 0:
                            break
                        f = get_denom_if_unit(rem_e)
                        if f is not None and f > e and f < best_max:
                            best_max = f
                            best_solution = [a, b, c, d, e, f]

    # Try 7-term solutions
    # a: 7/a > 5/31 => a < 217/5 = 43.4 => a <= 43
    if time.time() < deadline - 5.0: # Only if we have time to spare
        for a in range(7, 44):
            if time.time() > deadline:
                break
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
            b_max = min(best_max - 1, int(6 * rem_a.denominator / rem_a.numerator))
            
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    break
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    break
                c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator + 1)
                c_max = min(best_max - 1, int(5 * rem_b.denominator / rem_b.numerator))
                
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        break
                    rem_c = rem_b - Fraction(1, c)
                    if rem_c <= 0:
                        break
                    d_min = max(c + 1, (rem_c.denominator + rem_c.numerator - 1) // rem_c.numerator + 1)
                    d_max = min(best_max - 1, int(4 * rem_c.denominator / rem_c.numerator))
                    
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            break
                        rem_d = rem_c - Fraction(1, d)
                        if rem_d <= 0:
                            break
                        e_min = max(d + 1, (rem_d.denominator + rem_d.numerator - 1) // rem_d.numerator + 1)
                        e_max = min(best_max - 1, int(3 * rem_d.denominator / rem_d.numerator))
                        
                        for e in range(e_min, e_max + 1):
                            if time.time() > deadline:
                                break
                            rem_e = rem_d - Fraction(1, e)
                            if rem_e <= 0:
                                break
                            f_min = max(e + 1, (rem_e.denominator + rem_e.numerator - 1) // rem_e.numerator + 1)
                            f_max = min(best_max - 1, int(2 * rem_e.denominator / rem_e.numerator))
                            
                            for f in range(f_min, f_max + 1):
                                if time.time() > deadline:
                                    break
                                rem_f = rem_e - Fraction(1, f)
                                if rem_f <= 0:
                                    break
                                g = get_denom_if_unit(rem_f)
                                if g is not None and g > f and g < best_max:
                                    best_max = g
                                    best_solution = [a, b, c, d, e, f, g]

    # Final verification
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = [8, 31, 248]
            best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END
