import math
from fractions import Fraction
import time

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def check_time():
        return time.time() > deadline

    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248

    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == TARGET, f"Known solution invalid: {s}"

    # Structural change: instead of iterating over a and u (brute-force divisor search),
    # use a smarter enumeration. The key insight is that for 1/a + 1/b + 1/c = 5/31,
    # we can fix a and then the equation 1/b + 1/c = r where r = 5/31 - 1/a
    # becomes (pb - q)(pc - q) = q^2 where r = p/q.
    # 
    # But we can do better: iterate over possible c (the largest) from 10 to best_max-1,
    # and for each c, solve for a and b. Or better yet, use a two-level approach:
    # 
    # For 3-term: fix a (smallest), then for each a, the remaining is r = 5/31 - 1/a.
    # We need 1/b + 1/c = r with a < b < c < best_max.
    # This means b > a and c > b, so 1/b < r and 1/c < r.
    # Also b < 2/r (since 1/b > r/2) and c < 2/r.
    # And b > 1/r.
    # 
    # So for fixed a, b ranges over (a, 2/r) and c = 1/(r - 1/b).
    # We can iterate b and check if c is integer.
    #
    # But the divisor method is faster if we factor q^2. Let's use a hybrid:
    # For small a, q is manageable. Let's just use the divisor method but be smarter.

    # Actually, let me try a completely different approach: 
    # Iterate c from 10 to 247 (the largest denominator), and for each c,
    # we need 1/a + 1/b = 5/31 - 1/c = r_c where a < b < c.
    # r_c = p_c / q_c. Then (p_c*a - q_c)(p_c*b - q_c) = q_c^2.
    # For each divisor u of q_c^2, a = (u + q_c)/p_c, b = (q_c^2/u + q_c)/p_c.
    # Need a < b < c, all integers, and a >= 7 (since 5/31 < 1/6, smallest denom is 7).

    # This might be faster because q_c tends to be smaller for larger c.

    found_better = False

    for c in range(10, best_max):
        if check_time():
            break

        # r_c = 5/31 - 1/c
        r_c = TARGET - Fraction(1, c)
        if r_c <= 0:
            continue

        p = r_c.numerator
        q = r_c.denominator

        # (p*a - q)(p*b - q) = q^2
        # u = p*a - q, v = p*b - q, u*v = q^2
        # a = (u + q) / p, b = (v + q) / p
        # Need: a >= 7, a < b < c, a and b integers
        # a < b means u < v, so u < q (since u*v = q^2 and u < v implies u < q)
        # a >= 7: (u + q) / p >= 7 => u >= 7*p - q
        # b < c: (v + q) / p < c => v < c*p - q => q^2/u < c*p - q => u > q^2/(c*p - q)

        q2 = q * q

        u_min = max(1, 7 * p - q)
        if c * p <= q:
            continue  # No valid u since v = q^2/u >= q > c*p - q for all u <= q
        u_max = q - 1  # u < q to ensure a < b

        # Also u > q^2 / (c*p - q)
        if c * p > q:
            u_min = max(u_min, q2 // (c * p - q) + 1)

        if u_min > u_max:
            continue

        for u in range(u_min, u_max + 1):
            if check_time():
                break
            if q2 % u != 0:
                continue
            v = q2 // u
            num_a = u + q
            if num_a % p != 0:
                continue
            a = num_a // p
            if a < 7 or a >= c:
                continue
            num_b = v + q
            if num_b % p != 0:
                continue
            b = num_b // p
            if b <= a or b >= c:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
                    found_better = True

    # Now try to find solutions with even smaller max using DFS for 4+ terms
    if found_better:
        limit = best_max

        def search(remaining, used, min_d, depth, limit):
            if check_time():
                return None
            if remaining == 0:
                return []
            if depth >= MAX_TERMS:
                return None
            if remaining < 0:
                return None

            # Lower bound on next denominator
            # remaining = 1/d1 + ... + 1/dk, d_i >= min_d
            # Max possible sum with k terms starting at min_d: k / min_d
            # If remaining > k / min_d, infeasible
            # Min next denominator: 1/d <= remaining => d >= 1/remaining
            # Also need remaining - 1/d >= 0 => d >= 1/remaining
            # And d >= min_d
            min_d_eff = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            if min_d_eff >= limit:
                return None

            # Upper bound: we need at most (limit - 1) remaining terms, each >= min_d_eff
            # Max sum: (limit - min_d_eff) / min_d_eff ... this is complex, skip

            for d in range(min_d_eff, limit):
                if d in used:
                    continue
                if check_time():
                    return None

                rem_new = remaining - Fraction(1, d)
                if rem_new < 0:
                    break
                if rem_new == 0:
                    return [d]

                res = search(rem_new, used | {d}, d + 1, depth + 1, limit)
                if res is not None:
                    return [d] + res

            return None

        # Search for solutions with max < best_max using 4+ terms
        for a in range(7, limit):
            if check_time():
                break
            rem_a = TARGET - Fraction(1, a)
            if rem_a <= 0:
                continue
            res = search(rem_a, {a}, a + 1, 1, limit)
            if res is not None:
                candidate = [a] + res
                cand_max = max(candidate)
                if cand_max < best_max:
                    best_max = cand_max
                    best_solution = candidate

    return best_solution
