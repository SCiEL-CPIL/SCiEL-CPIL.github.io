"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50  # Safety margin before 60s kill

    # Try known small solutions first for speed
    known_solutions = [
        [5, 6, 7, 10, 14, 21, 35],
        [5, 6, 7, 8, 9, 10, 14],
        [5, 6, 7, 8, 9, 10, 15],
        [5, 6, 7, 8, 9, 10, 12],
        [5, 6, 7, 8, 9, 10, 18],
    ]
    
    for sol in known_solutions:
        s = Fraction(0, 1)
        for d in sol:
            s += Fraction(1, d)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol

    # Fast DFS search for k terms
    def fast_search(k, min_den=5):
        """Search for k distinct denominators >= min_den summing to 1."""
        target = Fraction(1, 1)
        found = [None]

        def dfs(start, remaining, chosen):
            if found[0] is not None:
                return
            if time.time() > deadline:
                return
            terms_left = k - len(chosen)
            if terms_left == 0:
                if remaining == 0:
                    found[0] = chosen[:]
                return
            if terms_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d > start and d >= 5 and d <= 10000000:
                            found[0] = chosen + [d]
                return

            # Pruning: max possible sum with terms_left terms starting from start+1
            max_possible = Fraction(0, 1)
            for i in range(terms_left):
                max_possible += Fraction(1, start + 1 + i)
            if max_possible < remaining:
                return

            # d_min: 1/d <= remaining => d >= ceil(1/remaining)
            d_min = max(start + 1, (1 / remaining).numerator // (1 / remaining).denominator + 1)

            # d_max: remaining <= terms_left / d => d <= terms_left / remaining
            d_max = int(terms_left / remaining)
            d_max = min(d_max, 10000000)

            if d_min > d_max:
                return

            for d in range(d_min, d_max + 1):
                if found[0] is not None or time.time() > deadline:
                    return
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                # Additional pruning: check if remaining can be filled
                # with (terms_left - 1) terms each > d
                if terms_left - 1 > 0:
                    min_next = Fraction(0, 1)
                    for i in range(terms_left - 1):
                        min_next += Fraction(1, d + 1 + i)
                    if min_next < new_remaining:
                        continue
                dfs(d, new_remaining, chosen + [d])

        dfs(min_den - 1, target, [])
        return found[0]

    # Try k=7 first (proven minimal), then increase
    for k in range(7, 11):
        result = fast_search(k)
        if result is not None:
            # Validate
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: DP over divisors of 2520 (guaranteed to find a solution)
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
