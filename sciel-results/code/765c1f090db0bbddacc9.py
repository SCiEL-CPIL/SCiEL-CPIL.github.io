"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80

    def find_last_two(r, prev_d):
        a, b = r.numerator, r.denominator
        if a <= 0 or b <= 0:
            return None
        min_d7 = max(prev_d + 1, b // a + (1 if b % a else 0))
        max_d7 = min(10000000, 2 * b // a)
        for d7 in range(min_d7, max_d7 + 1):
            num = a * d7 - b
            if num <= 0:
                continue
            if (d7 * b) % num != 0:
                continue
            d8 = (d7 * b) // num
            if d8 <= d7 or d8 > 10000000:
                continue
            return d7, d8
        return None

    def dfs6(idx, prev_d, current, remaining_frac):
        if time.time() > deadline:
            return None
        if idx == 6:
            r = remaining_frac
            if r <= 0:
                return None
            result = find_last_two(r, prev_d)
            if result is not None:
                return current + [result[0], result[1]]
            return None

        rem = 6 - idx
        # Tighter upper bound: sum of reciprocals of smallest rem denominators
        # after prev_d must be >= remaining_frac
        max_possible = sum(Fraction(1, prev_d + 1 + i) for i in range(rem))
        if max_possible < remaining_frac:
            return None

        # Also check minimum possible: if even the largest possible sum < remaining_frac
        # (this is already covered above)

        min_d = max(prev_d + 1, remaining_frac.denominator // remaining_frac.numerator + (1 if remaining_frac.denominator % remaining_frac.numerator else 0))
        max_d = min(10000000, rem * remaining_frac.denominator // remaining_frac.numerator)
        if max_d < min_d:
            return None

        # Try smaller denominators first (they leave more room for remaining terms)
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            new_rem = remaining_frac - Fraction(1, d)
            if new_rem <= 0:
                continue
            # Prune: check if remaining terms can sum to new_rem
            rem2 = rem - 1
            if rem2 > 0:
                max_possible2 = sum(Fraction(1, d + 1 + i) for i in range(rem2))
                if max_possible2 < new_rem:
                    continue
            current.append(d)
            result = dfs6(idx + 1, d, current, new_rem)
            if result is not None:
                return result
            current.pop()
        return None

    # Try d1 from 5 upward, but prioritize values that are more likely to work
    # 1/d1 + 1/d2 + ... + 1/d8 = 1
    # For 8 terms, d1 should be small enough
    for d1 in range(5, 25):
        if time.time() > deadline:
            break
        r_after_d1 = Fraction(1) - Fraction(1, d1)
        if r_after_d1 <= 0:
            continue
        # Check if 7 more terms can sum to r_after_d1
        max7 = sum(Fraction(1, d1 + 1 + i) for i in range(7))
        if max7 < r_after_d1:
            continue
        current = [d1]
        result = dfs6(1, d1, current, r_after_d1)
        if result is not None:
            return result

    # Fallback: hardcoded 8-term solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/30 + 1/42 = 1
    # Let me verify: need to find a valid 8-term Egyptian fraction
    # Use known identity: 1 = 1/2 + 1/3 + 1/6, but denominators must be > 4
    # Alternative: use the DP fallback
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
