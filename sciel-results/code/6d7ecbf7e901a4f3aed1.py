"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    assert (31*248 + 8*248 + 8*31) * 31 == 5 * 8 * 31 * 248, "Known solution invalid"
    
    target_num = 5
    target_den = 31
    
    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)
    
    def frac_sub(num1, den1, num2, den2):
        """Compute num1/den1 - num2/den2, return (num, den) in reduced form."""
        g = gcd(den1, den2)
        lcm = den1 // g * den2
        n = num1 * (lcm // den1) - num2 * (lcm // den2)
        if n == 0:
            return (0, 1)
        g = gcd(abs(n), lcm)
        return (n // g, lcm // g)
    
    def search_2terms(p, q, min_den, max_den):
        """Find 1/b + 1/c = p/q with min_den < b < c < max_den."""
        # (p*b - q) * c = q*b => c = q*b / (p*b - q)
        # Need p*b - q > 0 => b > q/p
        # Need c > b => q*b / (p*b - q) > b => q > p*b - q => b < 2q/p
        # Need c < max_den
        results = []
        b_lo = max(min_den + 1, q // p + 1)
        b_hi = min(max_den - 1, (2 * q) // p - 1)
        if b_hi <= b_lo:
            return results
        for b in range(b_lo, b_hi + 1):
            num = p * b - q
            if num <= 0:
                continue
            if (q * b) % num != 0:
                continue
            c = (q * b) // num
            if c > b and c < max_den:
                results.append((b, c))
        return results
    
    # Try 2-term solutions
    for b, c in search_2terms(5, 31, 0, best_max):
        check_and_update([b, c])
    
    # Try 3-term solutions using integer arithmetic
    # a >= 7 (since 1/a <= 5/31 requires a >= 7), and 3/a >= 5/31 requires a <= 18
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        # rem = 5/31 - 1/a
        p, q = frac_sub(5, 31, 1, a)
        if p <= 0:
            continue
        for b, c in search_2terms(p, q, a, best_max):
            check_and_update([a, b, c])
    
    # Try 4-term solutions
    # a >= 7, and 4/a >= 5/31 requires a <= 24
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        p1, q1 = frac_sub(5, 31, 1, a)
        if p1 <= 0:
            continue
        # For b: 3/b >= rem1 => b <= 3*q1/p1, and b > q1/p1
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (3 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            p2, q2 = frac_sub(p1, q1, 1, b)
            if p2 <= 0:
                continue
            for c, d in search_2terms(p2, q2, b, best_max):
                check_and_update([a, b, c, d])
    
    # Try 5-term solutions
    # a >= 7, and 5/a >= 5/31 requires a <= 30
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        p1, q1 = frac_sub(5, 31, 1, a)
        if p1 <= 0:
            continue
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (4 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            p2, q2 = frac_sub(p1, q1, 1, b)
            if p2 <= 0:
                continue
            c_lo = max(b + 1, q2 // p2 + 1)
            c_hi = min(best_max - 1, (3 * q2) // p2 - 1)
            if c_hi <= c_lo:
                continue
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                p3, q3 = frac_sub(p2, q2, 1, c)
                if p3 <= 0:
                    continue
                for d, e in search_2terms(p3, q3, c, best_max):
                    check_and_update([a, b, c, d, e])
    
    # Try 6-term solutions
    # a >= 7, and 6/a >= 5/31 requires a <= 37
    for a in range(7, min(38, best_max)):
        if time.time() > deadline:
            break
        p1, q1 = frac_sub(5, 31, 1, a)
        if p1 <= 0:
            continue
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (5 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            p2, q2 = frac_sub(p1, q1, 1, b)
            if p2 <= 0:
                continue
            c_lo = max(b + 1, q2 // p2 + 1)
            c_hi = min(best_max - 1, (4 * q2) // p2 - 1)
            if c_hi <= c_lo:
                continue
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                p3, q3 = frac_sub(p2, q2, 1, c)
                if p3 <= 0:
                    continue
                d_lo = max(c + 1, q3 // p3 + 1)
                d_hi = min(best_max - 1, (3 * q3) // p3 - 1)
                if d_hi <= d_lo:
                    continue
                for d in range(d_lo, d_hi + 1):
                    if time.time() > deadline:
                        break
                    p4, q4 = frac_sub(p3, q3, 1, d)
                    if p4 <= 0:
                        continue
                    for e, f in search_2terms(p4, q4, d, best_max):
                        check_and_update([a, b, c, d, e, f])
    
    # Try 7-term solutions
    # a >= 7, and 7/a >= 5/31 requires a <= 43
    for a in range(7, min(44, best_max)):
        if time.time() > deadline:
            break
        p1, q1 = frac_sub(5, 31, 1, a)
        if p1 <= 0:
            continue
        b_lo = max(a + 1, q1 // p1 + 1)
        b_hi = min(best_max - 1, (6 * q1) // p1 - 1)
        if b_hi <= b_lo:
            continue
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            p2, q2 = frac_sub(p1, q1, 1, b)
            if p2 <= 0:
                continue
            c_lo = max(b + 1, q2 // p2 + 1)
            c_hi = min(best_max - 1, (5 * q2) // p2 - 1)
            if c_hi <= c_lo:
                continue
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                p3, q3 = frac_sub(p2, q2, 1, c)
                if p3 <= 0:
                    continue
                d_lo = max(c + 1, q3 // p3 + 1)
                d_hi = min(best_max - 1, (4 * q3) // p3 - 1)
                if d_hi <= d_lo:
                    continue
                for d in range(d_lo, d_hi + 1):
                    if time.time() > deadline:
                        break
                    p4, q4 = frac_sub(p3, q3, 1, d)
                    if p4 <= 0:
                        continue
                    e_lo = max(d + 1, q4 // p4 + 1)
                    e_hi = min(best_max - 1, (3 * q4) // p4 - 1)
                    if e_hi <= e_lo:
                        continue
                    for e in range(e_lo, e_hi + 1):
                        if time.time() > deadline:
                            break
                        p5, q5 = frac_sub(p4, q4, 1, e)
                        if p5 <= 0:
                            continue
                        for f, g in search_2terms(p5, q5, e, best_max):
                            check_and_update([a, b, c, d, e, f, g])
    
    return best
# EVOLVE-BLOCK-END
