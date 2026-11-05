"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    def try_find_solution(n_terms, max_den=10000000):
        """Find n_terms distinct denominators in [5, max_den] summing to 1."""
        if time.time() > deadline:
            return None

        def dfs(remaining, n_left, min_den, current):
            if time.time() > deadline:
                return None

            if n_left == 0:
                if remaining == 0:
                    return current[:]
                return None

            if n_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        x = den // num
                        if x > min_den and x <= max_den:
                            return current + [x]
                return None

            if remaining <= 0:
                return None

            lower_a = max(min_den, int(1 / float(remaining)) + 1)
            upper_a = min(max_den, int(n_left / float(remaining)) + 1)

            if lower_a > upper_a:
                return None

            for a in range(lower_a, upper_a + 1):
                if time.time() > deadline:
                    return None

                max_possible = 0.0
                for i in range(n_left):
                    max_possible += 1.0 / (a + i)
                if max_possible < float(remaining) - 1e-15:
                    break

                new_remaining = remaining - Fraction(1, a)
                if new_remaining <= 0:
                    continue

                result = dfs(new_remaining, n_left - 1, a + 1, current + [a])
                if result is not None:
                    return result

            return None

        return dfs(Fraction(1), n_terms, 5, [])

    # 8 terms impossible (max sum < 1), so try 9 first
    for n_terms in range(9, 20):
        if time.time() > deadline - 5:
            break

        sol = try_find_solution(n_terms, max_den=10000000)
        if sol is not None:
            s = sum(Fraction(1, x) for x in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
                return sol

    # Fallback: use the DP approach
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            break

    if target in reached:
        sol = reached[target]
        s = sum(Fraction(1, x) for x in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
            return sol

    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
