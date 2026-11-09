"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    Uses divisor-based enumeration for 3 terms and bounded search for 4+ terms.
    """
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution: [8, 31, 248], max = 248
    # Verify: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31 ✓
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    def find_3_terms_exhaustive(max_c):
        """
        Exhaustively check all 3-term solutions 1/a+1/b+1/c = 5/31 with a<b<c<=max_c.
        Uses divisor structure: fix c, compute rem = 5/31 - 1/c, then find a,b.
        """
        nonlocal best_max_den, best_solution
        if time.time() > deadline:
            return False
        
        for c in range(247, max_c, -1):  # Check c from 247 down to max_c
            if time.time() > deadline:
                return False
            rem = target - Fraction(1, c)
            if rem <= 0:
                continue
            # 1/a + 1/b = rem = p/q
            # => a+b = p*a*b/q => q(a+b) = p*a*b
            # => p*a*b - q*a - q*b = 0
            # => (p*a - q)(p*b - q) = q^2
            p = rem.numerator
            q = rem.denominator
            # Need (p*a - q) and (p*b - q) to be positive divisors of q^2
            # with p*a - q > 0 => a > q/p, and a < b < c
            # Also a >= 7 (since 1/a < 5/31 requires a > 31/5 = 6.2)
            
            # Find all divisor pairs of q*q
            q2 = q * q
            divisors = []
            i = 1
            while i * i <= q2:
                if q2 % i == 0:
                    divisors.append(i)
                    if i != q2 // i:
                        divisors.append(q2 // i)
                i += 1
            
            for d1 in divisors:
                if time.time() > deadline:
                    return False
                d2 = q2 // d1
                # p*a - q = d1 => a = (d1 + q) / p
                # p*b - q = d2 => b = (d2 + q) / p
                if (d1 + q) % p != 0 or (d2 + q) % p != 0:
                    continue
                a = (d1 + q) // p
                b = (d2 + q) // p
                # Ensure a < b < c and a >= 7
                if a < 7 or a >= b or b >= c:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    sol = [a, b, c]
                    if max(sol) < best_max_den:
                        best_max_den = max(sol)
                        best_solution = sol
                    return True
        return False
    
    # Check if any 3-term solution exists with max < 248
    # We already know [8,31,248] works. Check c from 247 down to 7.
    # For efficiency, only check c where 5/31 - 1/c > 0, i.e., c > 31/5 = 6.2
    found_better_3 = find_3_terms_exhaustive(7)
    
    # If no better 3-term found, try 4-term search with max < 248
    if best_max_den == 248:
        def find_4_terms_bounded(max_d):
            """
            Find 1/a+1/b+1/c+1/d = 5/31 with 7<=a<b<c<d<=max_d.
            Uses meet-in-the-middle: precompute pair sums.
            """
            nonlocal best_max_den, best_solution
            if time.time() > deadline:
                return False
            
            # Precompute all valid (a,b) pairs with 7<=a<b<=max_d
            # where 1/a + 1/b < 5/31
            pair_sums = {}
            for a in range(7, max_d + 1):
                if time.time() > deadline:
                    return False
                # 1/a < 5/31 => a > 31/5, so a >= 7 is fine
                # 1/a + 1/b < 5/31 => 1/b < 5/31 - 1/a
                rem_after_a = target - Fraction(1, a)
                if rem_after_a <= 0:
                    continue
                # b > 1/rem_after_a
                b_min = max(a + 1, (rem_after_a.denominator + rem_after_a.numerator - 1) // rem_after_a.numerator + 1)
                # Also need 1/a + 1/b < 5/31, which is already ensured by rem_after_a > 0 and b > 1/rem_after_a
                # But also 1/b < rem_after_a => b > 1/rem_after_a
                for b in range(b_min, max_d + 1):
                    if time.time() > deadline:
                        return False
                    s = Fraction(1, a) + Fraction(1, b)
                    if s >= target:
                        break
                    key = s
                    if key not in pair_sums:
                        pair_sums[key] = []
                    pair_sums[key].append((a, b))
            
            # Now for each (c,d) pair with c<d<=max_d, check if target - 1/c - 1/d is in pair_sums
            # and the (a,b) pair satisfies b < c
            for c in range(8, max_d + 1):  # c >= 8 since a>=7, b>=8, c>=9 actually
                if time.time() > deadline:
                    return False
                rem_cd = target - Fraction(1, c)
                if rem_cd <= 0:
                    continue
                d_min = max(c + 1, (rem_cd.denominator + rem_cd.numerator - 1) // rem_cd.numerator + 1)
                for d in range(d_min, max_d + 1):
                    if time.time() > deadline:
                        return False
                    rem = target - Fraction(1, c) - Fraction(1, d)
                    if rem <= 0:
                        break
                    if rem in pair_sums:
                        for (a, b) in pair_sums[rem]:
                            if b < c:
                                sol = [a, b, c, d]
                                if max(sol) < best_max_den:
                                    best_max_den = max(sol)
                                    best_solution = sol
                                return True
            return False
        
        # Search 4-term solutions with max < 248
        for md in range(247, 20, -1):
            if time.time() > deadline:
                break
            if find_4_terms_bounded(md):
                if best_max_den <= md:
                    break
    
    # Try 5-term search if still 248
    if best_max_den == 248 and time.time() < deadline - 10:
        def find_5_terms_bounded(max_e):
            """
            Find 5-term solution with max <= max_e.
            Fix (a,b), then solve 3-term for remainder.
            """
            nonlocal best_max_den, best_solution
            if time.time() > deadline:
                return False
            
            for a in range(7, max_e + 1):
                if time.time() > deadline:
                    return False
                rem_a = target - Fraction(1, a)
                if rem_a <= 0:
                    continue
                b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
                for b in range(b_min, max_e + 1):
                    if time.time() > deadline:
                        return False
                    rem_ab = rem_a - Fraction(1, b)
                    if rem_ab <= 0:
                        break
                    # Solve 1/c + 1/d + 1/e = rem_ab with b < c < d < e <= max_e
                    # c > 1/rem_ab, and 3/c >= rem_ab => c <= 3/rem_ab
                    c_lo = max(b + 1, (rem_ab.denominator + rem_ab.numerator - 1) // rem_ab.numerator + 1)
                    c_hi = min(max_e - 2, 3 * rem_ab.denominator // rem_ab.numerator)
                    if c_lo > c_hi:
                        continue
                    for c in range(c_lo, c_hi + 1):
                        if time.time() > deadline:
                            return False
                        rem_bc = rem_ab - Fraction(1, c)
                        if rem_bc <= 0:
                            continue
                        # 1/d + 1/e = rem_bc, d > c, e > d
                        # e = d / (rem_bc * d - 1) in fraction form
                        p = rem_bc.numerator
                        q = rem_bc.denominator
                        # d > 1/rem_bc => d > q/p
                        d_lo = max(c + 1, q // p + 1)
                        # e > d => d/(p*d-q) > d => 1 > p*d-q => d < (q+1)/p... 
                        # Actually: e = d*q/(p*d-q) > d => q > p*d-q => 2q > p*d => d < 2q/p
                        d_hi = min(max_e - 1, (2 * q) // p)
                        if d_lo > d_hi:
                            continue
                        for d in range(d_lo, d_hi + 1):
                            if time.time() > deadline:
                                return False
                            denom_e = p * d - q
                            if denom_e <= 0:
                                continue
                            if (d * q) % denom_e != 0:
                                continue
                            e = (d * q) // denom_e
                            if e <= d or e > max_e:
                                continue
                            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == target:
                                sol = [a, b, c, d, e]
                                if max(sol) < best_max_den:
                                    best_max_den = max(sol)
                                    best_solution = sol
                                return True
            return False
        
        for me in range(247, 25, -1):
            if time.time() > deadline:
                break
            if find_5_terms_bounded(me):
                if best_max_den <= me:
                    break
    
    # Final validation
    total = sum(Fraction(1, d) for d in best_solution)
    if total != target or len(set(best_solution)) != len(best_solution):
        # Fallback to known valid solution
        best_solution = [8, 31, 248]
    
    return best_solution
# EVOLVE-BLOCK-END
