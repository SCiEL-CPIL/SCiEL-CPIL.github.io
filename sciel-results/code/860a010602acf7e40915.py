"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd, isqrt

def construct():
    deadline = time.time() + 55.0
    target = Fraction(5, 31)
    
    # Known fallback
    best_max = 248
    best_solution = [8, 31, 248]
    
    def solve_pair(p, q, lo=1, hi=None):
        """Find all pairs (a,b) with lo <= a < b <= hi such that 1/a + 1/b = p/q.
        Uses the identity: (qa-p)(qb-p) = p^2
        Returns list of (a,b) tuples."""
        if p <= 0 or q <= 0:
            return []
        g = gcd(p, q)
        p_r, q_r = p // g, q // g
        p2 = p_r * p_r
        sqrt_p2 = isqrt(p2)
        results = []
        for u in range(1, sqrt_p2 + 1):
            if p2 % u != 0:
                continue
            v = p2 // u
            # (qa-p) = u, (qb-p) = v  OR  (qa-p) = v, (qb-p) = u
            for u1, u2 in [(u, v), (v, u)]:
                if (u1 + p_r) % q_r != 0:
                    continue
                a = (u1 + p_r) // q_r
                if (u2 + p_r) % q_r != 0:
                    continue
                b = (u2 + p_r) // q_r
                if a < 1 or b < 1:
                    continue
                if a > b:
                    a, b = b, a
                if a == b:
                    continue
                if a < lo:
                    continue
                if hi is not None and b > hi:
                    continue
                results.append((a, b))
        return results

    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each c, solve 1/a + 1/b = 5/31 - 1/c = (5c-31)/(31c)
    # a must be >= 7 (since 1/a < 5/31 means a > 6.2)
    # b > a, c > b
    for c in range(8, 248):
        if time.time() > deadline:
            break
        p = 5 * c - 31
        q = 31 * c
        if p <= 0:
            continue
        pairs = solve_pair(p, q, lo=7, hi=c - 1)
        for (a, b) in pairs:
            if a < b < c:
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    if c < best_max:
                        best_max = c
                        best_solution = [a, b, c]
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # For each d, solve 1/a + 1/b + 1/c = (5d-31)/(31d)
    # For each c, solve 1/a + 1/b = (5d-31)/(31d) - 1/c
    for d in range(9, best_max):
        if time.time() > deadline:
            break
        p_d = 5 * d - 31
        q_d = 31 * d
        if p_d <= 0:
            continue
        # For each c from 8 to d-1, solve 1/a+1/b+1/c = p_d/q_d
        # i.e., 1/a + 1/b = p_d/q_d - 1/c = (p_d*c - q_d)/(q_d*c)
        for c in range(8, d):
            if time.time() > deadline:
                break
            p_c = p_d * c - q_d
            q_c = q_d * c
            if p_c <= 0:
                continue
            pairs = solve_pair(p_c, q_c, lo=7, hi=c - 1)
            for (a, b) in pairs:
                if a < b < c < d:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                        if d < best_max:
                            best_max = d
                            best_solution = [a, b, c, d]
    
    # Try 5-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # For each e, solve 4-term = (5e-31)/(31e)
    for e in range(10, best_max):
        if time.time() > deadline:
            break
        p_e = 5 * e - 31
        q_e = 31 * e
        if p_e <= 0:
            continue
        # For each d, solve 3-term = p_e/q_e - 1/d
        for d in range(9, e):
            if time.time() > deadline:
                break
            p_d = p_e * d - q_e
            q_d = q_e * d
            if p_d <= 0:
                continue
            # For each c, solve 2-term = p_d/q_d - 1/c
            for c in range(8, d):
                if time.time() > deadline:
                    break
                p_c = p_d * c - q_d
                q_c = q_d * c
                if p_c <= 0:
                    continue
                pairs = solve_pair(p_c, q_c, lo=7, hi=c - 1)
                for (a, b) in pairs:
                    if a < b < c < d < e:
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == target:
                            if e < best_max:
                                best_max = e
                                best_solution = [a, b, c, d, e]
    
    # Try 6-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 5/31
    for f in range(11, best_max):
        if time.time() > deadline:
            break
        p_f = 5 * f - 31
        q_f = 31 * f
        if p_f <= 0:
            continue
        for e in range(10, f):
            if time.time() > deadline:
                break
            p_e = p_f * e - q_f
            q_e = q_f * e
            if p_e <= 0:
                continue
            for d in range(9, e):
                if time.time() > deadline:
                    break
                p_d = p_e * d - q_e
                q_d = q_e * d
                if p_d <= 0:
                    continue
                for c in range(8, d):
                    if time.time() > deadline:
                        break
                    p_c = p_d * c - q_d
                    q_c = q_d * c
                    if p_c <= 0:
                        continue
                    pairs = solve_pair(p_c, q_c, lo=7, hi=c - 1)
                    for (a, b) in pairs:
                        if a < b < c < d < e < f:
                            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) + Fraction(1, f) == target:
                                if f < best_max:
                                    best_max = f
                                    best_solution = [a, b, c, d, e, f]
    
    # Final verification
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = [8, 31, 248]
            best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END
