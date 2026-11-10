"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80

    def dfs(remaining, min_denom, max_terms, current):
        if time.time() > deadline:
            return None
        if max_terms == 0:
            return current if remaining == 0 else None
        if remaining == 0:
            return current
        if remaining < 0:
            return None

        # Max possible sum with max_terms terms from min_denom
        max_possible = sum(Fraction(1, min_denom + i) for i in range(max_terms))
        if max_possible < remaining:
            return None

        # Lower bound: d >= ceil(1/remaining) and d >= min_denom
        if remaining > 0:
            lower_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        else:
            lower_d = 1
        lower_d = max(lower_d, min_denom, 5)

        # Upper bound: 1/d <= remaining => d >= 1/remaining (already handled)
        # Also need remaining - 1/d >= 0
        # Practical upper bound
        upper_d = min(lower_d + max_terms * 500, 10000000)

        for d in range(lower_d, upper_d + 1):
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                continue
            if max_terms - 1 > 0:
                max_next = sum(Fraction(1, d + 1 + i) for i in range(max_terms - 1))
                if max_next < new_rem:
                    break
            result = dfs(new_rem, d + 1, max_terms - 1, current + [d])
            if result is not None:
                return result
        return None

    for num_terms in range(5, 25):
        max_sum = sum(Fraction(1, 5 + i) for i in range(num_terms))
        if max_sum < 1:
            continue
        result = dfs(Fraction(1, 1), 5, num_terms, [])
        if result is not None:
            s = sum(Fraction(1, x) for x in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= x <= 10000000 for x in result):
                return result

    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
