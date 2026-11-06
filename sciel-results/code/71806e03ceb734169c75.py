"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 55
    target = Fraction(5, 31)
    
    # Fallback known solution
    best_max = 248
    best_solution = [8, 31, 248]
    
    # Strategy: Search for minimum maximum denominator M
    # Try 2 terms first (we know it's impossible, but verify quickly)
    # Then 3 terms, then 4 terms, etc.
    # For each term count, search for minimum max denominator
    
    # Helper: solve 1/a + 1/b = p/q for integers a, b with a < b
    # (qa-p)(qb-p) = p^2
    def solve_two_term(p, q, a_min, b_max):
        """Find pairs (a,b) with a_min <= a < b <= b_max such that 1/a + 1/b = p/q."""
        if p <= 0 or q <= 0:
            return []
        g = gcd(p, q)
        p_r, q_r = p // g, q // g
        # (q_r * a - p_r)(q_r * b - p_r) = p_r^2
        p2 = p_r * p_r
        results = []
        # Find divisors of p2
        # u = q_r * a - p_r, so u must be a divisor of p2
        # a = (u + p_r) / q_r must be integer
        # u > 0 (since a >= a_min >= 1, and q_r*a > p_r for valid fractions)
        # Actually u can be negative if a is small
        # Let's iterate over divisors
        # For efficiency, iterate u from 1 to sqrt(p2) and check both u and p2//u
        import math
        sqrt_p2 = int(math.isqrt(p2))
        seen_pairs = set()
        for u in range(1, sqrt_p2 + 1):
            if p2 % u != 0:
                continue
            for pair in [(u, p2 // u), (p2 // u, u)]:
                u1, u2 = pair
                # a = (u1 + p_r) / q_r
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
                if a < a_min or b > b_max:
                    continue
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                results.append((a, b))
        return results
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each c from 7 upward, solve 1/a + 1/b = 5/31 - 1/c
    for c in range(7, 300):
        if time.time() > deadline:
            break
        # 1/a + 1/b = 5/31 - 1/c = (5c - 31) / (31c)
        p = 5 * c - 31
        q = 31 * c
        if p <= 0:
            continue
        # Find a, b with 7 <= a < b < c (since a < b < c)
        # Actually a can be as small as 7 (since 1/7 < 5/31)
        # Wait, 1/7 ≈ 0.143 < 0.161 = 5/31, so a >= 7
        # But we need a < b < c, so b < c
        pairs = solve_two_term(p, q, 7, c - 1)
        for (a, b) in pairs:
            if a < b and b < c:
                # Verify
                s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                if s == target:
                    if c < best_max:
                        best_max = c
                        best_solution = [a, b, c]
                    break
        if best_max == c:
            # Can't do better with 3 terms
            break
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # For each d from 7 upward, solve 1/a + 1/b + 1/c = 5/31 - 1/d
    # This is more complex. Let's use meet-in-the-middle:
    # Precompute all sums 1/a + 1/b for a < b with a, b <= d
    # Then check if 5/31 - 1/d - (1/a + 1/b) = 1/c for some c
    
    # Actually, let's be smarter. For 4 terms with max d:
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Fix d, then we need 1/a + 1/b + 1/c = 5/31 - 1/d = (5d-31)/(31d)
    # For each c < d, solve 1/a + 1/b = (5d-31)/(31d) - 1/c = ((5d-31)c - 31d) / (31dc)
    
    for d in range(7, best_max):
        if time.time() > deadline:
            break
        P = 5 * d - 31
        Q = 31 * d
        if P <= 0:
            continue
        # For each c from 7 to d-1:
        for c in range(7, d):
            if time.time() > deadline:
                break
            # 1/a + 1/b = P/Q - 1/c = (Pc - Q) / (Qc)
            p = P * c - Q
            q = Q * c
            if p <= 0:
                continue
            pairs = solve_two_term(p, q, 7, c - 1)
            for (a, b) in pairs:
                if a < b and b < c:
                    s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d)
                    if s == target:
                        if d < best_max:
                            best_max = d
                            best_solution = [a, b, c, d]
                        break
            if best_max == d:
                break
        if best_max == d:
            break
    
    # Try 5-term solutions if needed
    if best_max > 50:  # Only try if we haven't found something good
        for e in range(7, min(best_max, 100)):
            if time.time() > deadline:
                break
            P = 5 * e - 31
            Q = 31 * e
            if P <= 0:
                continue
            # Need 1/a + 1/b + 1/c + 1/d = P/Q with a < b < c < d < e
            # This is getting expensive. Use meet-in-the-middle for pairs.
            # Precompute all 1/a + 1/b for 7 <= a < b < e
            pair_sums = defaultdict(list)
            for a in range(7, e - 1):
                for b in range(a + 1, e):
                    s = Fraction(1, a) + Fraction(1, b)
                    pair_sums[s].append((a, b))
            
            # For each c, d with c < d < e:
            # 1/a + 1/b = P/Q - 1/c - 1/d
            for c in range(7, e - 1):
                if time.time() > deadline:
                    break
                for d in range(c + 1, e):
                    if time.time() > deadline:
                        break
                    rem = P / Q - Fraction(1, c) - Fraction(1, d)
                    if rem <= 0:
                        continue
                    if rem in pair_sums:
                        for (a, b) in pair_sums[rem]:
                            if a < b and b < c:
                                s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e)
                                if s == target:
                                    if e < best_max:
                                        best_max = e
                                        best_solution = [a, b, c, d, e]
                                    break
                    if best_max == e:
                        break
                if best_max == e:
                    break
            if best_max == e:
                break
    
    # Final verification
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = [8, 31, 248]
            best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END
