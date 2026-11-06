"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == target, "Known solution invalid"
    
    def is_unit_fraction(frac):
        """Check if frac is 1/k for some integer k."""
        if frac.numerator == 1:
            return frac.denominator
        return None

    def update_best(denoms):
        nonlocal best, best_max
        m = max(denoms)
        if m < best_max:
            # Verify distinctness and range
            if len(set(denoms)) == len(denoms) and all(1 <= d <= 10000000 for d in denoms):
                # Verify sum
                if sum(Fraction(1, d) for d in denoms) == target:
                    best_max = m
                    best = list(denoms)
                    return True
        return False

    # Meet-in-the-middle: precompute all sums of 1 unit fraction (trivially)
    # and all sums of 2 unit fractions with denominators up to some bound
    # Then for target = sum of left + sum of right, check if they combine
    
    # Strategy: try to find solutions with max denominator < 248
    # Precompute all pairs (a,b) with a < b, both <= 200, and store their sum
    # Then for each pair, check if target - sum(pair) is a sum of 1, 2, or 3 unit fractions
    # with denominators > b and < 248
    
    # Better: precompute all sums of up to 2 unit fractions from denominators in [7, 200]
    # Then for each such sum S, check if target - S can be written as sum of up to 3 unit fractions
    # with denominators > max(pair) and < 248
    
    # Let's do: precompute all sums of exactly 1 and exactly 2 unit fractions
    # from denominators in a reasonable range, store in dict: sum -> list of denom tuples
    
    # For efficiency, let's focus on finding solutions where the largest denom is < 248
    # Try denominators up to 247
    
    # Precompute sums of 1 unit fraction: {1/d: (d,)}
    # Precompute sums of 2 unit fractions: {1/a + 1/b: (a,b)} for a < b
    # Precompute sums of 3 unit fractions: {1/a + 1/b + 1/c: (a,b,c)} for a < b < c
    
    # Then target = sum1 + sum2, or sum1 + sum3, or sum2 + sum2, etc.
    # But we need all denominators distinct.
    
    # Let's try: for total terms = 3, 4, or 5
    # For 3 terms: just do the direct search (already doing this)
    # For 4 terms: split as 2+2. Precompute all pair sums, then for each pair sum S1,
    #   check if target - S1 is also a pair sum, with all 4 denominators distinct.
    # For 5 terms: split as 2+3 or 1+4. Precompute pair sums and triple sums.
    
    # Let's do the 2+2 meet-in-the-middle for 4-term solutions
    max_den_search = best_max - 1  # we want max < best_max
    
    # Precompute all pair sums with denominators in [2, max_den_search]
    # This could be large, so let's bound it
    # 1/a + 1/b >= 2/max_den_search, and we need 1/a + 1/b <= target
    # So a,b must be at least 1/target * 2 = 62/5 = 12.4, so a,b >= 13? No...
    # 1/a + 1/b <= 5/31 => if a=b, 2/a <= 5/31 => a >= 62/5 = 12.4, so a >= 13
    # But a and b can be different.
    
    # Let's just iterate a from 7 to max_den_search, b from a+1 to max_den_search
    # and store sums
    
    pair_sums = {}  # Fraction -> list of (a, b)
    
    # Bound: 1/a + 1/b must be <= target and > 0
    # Also 1/a + 1/b must be such that target - (1/a + 1/b) is positive and can be
    # written as 1/c + 1/d with c > b, d > c
    # target - (1/a+1/b) >= 1/(max_den_search) + 1/(max_den_search) = 2/max_den_search
    # So 1/a + 1/b <= target - 2/max_den_search
    
    # Let's just build the table for a in [7, 200], b in [a+1, 200]
    # That's about 193*192/2 ≈ 18k entries, very fast
    
    upper_bound = min(200, max_den_search)
    
    for a in range(7, upper_bound + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, upper_bound + 1):
            s = Fraction(1, a) + Fraction(1, b)
            if s >= target:
                break  # b increasing makes s decrease... wait no, b increasing makes 1/b decrease
            # Actually as b increases, 1/b decreases, so s decreases. 
            # We want s < target. If s >= target, we need larger b.
            # But we're iterating b increasing, so s is decreasing.
            # If s >= target at some b, we continue to larger b where s gets smaller.
            # Actually we should skip if s >= target since we need target - s > 0.
            if s >= target:
                continue
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((a, b))
    
    # Now for each pair sum S in pair_sums, check if target - S is also in pair_sums
    # with all 4 denominators distinct
    for S, pairs1 in pair_sums.items():
        if time.time() > deadline:
            break
        rem = target - S
        if rem <= 0:
            continue
        if rem in pair_sums:
            pairs2 = pair_sums[rem]
            for (a, b) in pairs1:
                for (c, d) in pairs2:
                    # Check all distinct: a, b, c, d all different
                    denoms = [a, b, c, d]
                    if len(set(denoms)) == 4:
                        if update_best(denoms):
                            pass
    
    # Also try 1+3 split for 4-term solutions
    # Precompute triple sums
    triple_sums = {}  # Fraction -> list of (a, b, c)
    
    for a in range(7, min(100, upper_bound) + 1):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        for b in range(a + 1, min(150, upper_bound) + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            # c must satisfy 1/c = rem_b, so c = 1/rem_b
            c = is_unit_fraction(rem_b)
            if c is not None and c > b and c <= upper_bound:
                s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                if s not in triple_sums:
                    triple_sums[s] = []
                triple_sums[s].append((a, b, c))
    
    # Now for 1+3: for each 1/d, check if target - 1/d is in triple_sums
    for d in range(7, upper_bound + 1):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, d)
        if rem <= 0:
            continue
        if rem in triple_sums:
            for (a, b, c) in triple_sums[rem]:
                denoms = [d, a, b, c]
                if len(set(denoms)) == 4:
                    if update_best(denoms):
                        pass
    
    # Also try the original direct search but with tighter bounds
    # Search for 3-term solutions
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = a + 1
        b_max = int(2 / rem_a) + 1
        for b in range(b_min, min(b_max, best_max)):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c = is_unit_fraction(rem_b)
            if c is not None and c > b and c < best_max:
                if update_best([a, b, c]):
                    pass
    
    # Search for 4-term solutions directly (in case meet-in-middle missed some)
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = a + 1
        b_max = int(3 / rem_a) + 1
        for b in range(b_min, min(b_max, best_max)):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_min = b + 1
            c_max = int(2 / rem_b) + 1
            for c in range(c_min, min(c_max, best_max)):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d = is_unit_fraction(rem_c)
                if d is not None and d > c and d < best_max:
                    if update_best([a, b, c, d]):
                        pass
    
    # Search for 5-term solutions
    for a in range(7, 31):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = a + 1
        b_max = int(4 / rem_a) + 1
        for b in range(b_min, min(b_max, best_max)):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            c_min = b + 1
            c_max = int(3 / rem_b) + 1
            for c in range(c_min, min(c_max, best_max)):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                d_min = c + 1
                d_max = int(2 / rem_c) + 1
                for d in range(d_min, min(d_max, best_max)):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        continue
                    e = is_unit_fraction(rem_d)
                    if e is not None and e > d and e < best_max:
                        if update_best([a, b, c, d, e]):
                            pass
    
    return best
# EVOLVE-BLOCK-END
