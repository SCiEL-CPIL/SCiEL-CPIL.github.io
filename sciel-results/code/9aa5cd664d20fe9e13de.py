"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    target = Fraction(5, 31)
    
    best_solution = [7, 62, 434]
    best_max_den = 434
    
    # Search for better 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    for a in range(7, 200):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            break
        p, q = r1.numerator, r1.denominator
        # 1/b + 1/c = p/q => c = b*q / (p*b - q)
        # c > b => b < 2q/p
        b_max = min(int(2 * q // p) + 1, best_max_den)
        for b in range(a + 1, b_max):
            if time.time() > deadline:
                break
            denom = p * b - q
            if denom <= 0:
                continue
            if (b * q) % denom != 0:
                continue
            c = (b * q) // denom
            if c > b and c < best_max_den and c <= 10000000:
                if a != b and b != c and a != c:
                    best_max_den = c
                    best_solution = [a, b, c]
                    break
    
    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # For each (a,b,c), solve for d: 1/d = 5/31 - 1/a - 1/b - 1/c
    # d must be integer > c, and d < best_max_den
    for a in range(7, 30):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            break
        # 1/b + 1/c + 1/d = r1
        # b > a, and 1/b < r1 => b > 1/r1
        # Also 3/b <= r1 => b >= 3/r1 (rough lower bound)
        # Upper bound: b < 1/r1 (since 1/b < r1)
        # More tightly: 1/b < r1 and 2/c + 2/d < r1 - 1/b... 
        # Simple bound: b in (a+1, 1/r1)
        b_lo = a + 1
        b_hi = int(1 / r1) + 1  # b must satisfy 1/b < r1
        b_hi = min(b_hi, best_max_den)
        for b in range(b_lo, b_hi):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            # 1/c + 1/d = r2, c > b, d > c
            # c < 2/r2 (for d > c)
            c_lo = b + 1
            c_hi = int(2 / r2) + 1
            c_hi = min(c_hi, best_max_den)
            for c in range(c_lo, c_hi):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                # d = 1/r3, must be integer
                if r3.denominator == 1:
                    d = r3.numerator
                    if d > c and d < best_max_den and d <= 10000000:
                        if len({a, b, c, d}) == 4:
                            best_max_den = d
                            best_solution = [a, b, c, d]
                elif r3.numerator % r3.denominator == 0:
                    d = r3.numerator // r3.denominator
                    if d > c and d < best_max_den and d <= 10000000:
                        if len({a, b, c, d}) == 4:
                            best_max_den = d
                            best_solution = [a, b, c, d]
    
    # Search for 5-term solutions if time permits
    if time.time() < deadline - 20:
        for a in range(7, 20):
            if time.time() > deadline:
                break
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                break
            b_lo = a + 1
            b_hi = min(int(1 / r1) + 1, best_max_den)
            for b in range(b_lo, b_hi):
                if time.time() > deadline:
                    break
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                c_lo = b + 1
                c_hi = min(int(1 / r2) + 1, best_max_den)
                for c in range(c_lo, c_hi):
                    if time.time() > deadline:
                        break
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0:
                        continue
                    d_lo = c + 1
                    d_hi = min(int(1 / r3) + 1, best_max_den)
                    for d in range(d_lo, d_hi):
                        if time.time() > deadline:
                            break
                        r4 = r3 - Fraction(1, d)
                        if r4 <= 0:
                            continue
                        # e = 1/r4 must be integer
                        if r4.denominator == 1:
                            e = r4.numerator
                            if e > d and e < best_max_den and e <= 10000000:
                                if len({a, b, c, d, e}) == 5:
                                    best_max_den = e
                                    best_solution = [a, b, c, d, e]
                        elif r4.numerator % r4.denominator == 0:
                            e = r4.numerator // r4.denominator
                            if e > d and e < best_max_den and e <= 10000000:
                                if len({a, b, c, d, e}) == 5:
                                    best_max_den = e
                                    best_solution = [a, b, c, d, e]
    
    return best_solution
# EVOLVE-BLOCK-END
