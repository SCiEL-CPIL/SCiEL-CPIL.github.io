from fractions import Fraction
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def find_solution_greedy():
        sol = []
        rem = Fraction(1, 1)
        min_d = 5
        while rem > 0 and len(sol) < 256:
            d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            if d > 10000000:
                break
            sol.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
        return sol if rem == 0 else None

    baseline = find_solution_greedy()
    if not baseline:
        return []

    best_sol = baseline
    best_count = len(baseline)

    def dfs(rem_num, rem_den, min_d, current, remaining_terms):
        nonlocal best_sol, best_count

        if time.time() > deadline:
            return

        if rem_num == 0:
            if len(current) < best_count:
                best_count = len(current)
                best_sol = current[:]
            return

        if rem_num < 0 or rem_den <= 0:
            return

        if len(current) + remaining_terms >= best_count:
            return

        # Check feasibility: max sum with remaining_terms terms >= min_d
        max_sum_num = 0
        max_sum_den = 1
        for i in range(remaining_terms):
            d = min_d + i
            max_sum_num = max_sum_num * d + max_sum_den
            max_sum_den = max_sum_den * d

        if rem_num * max_sum_den > max_sum_num * rem_den:
            return

        d_start = max(min_d, (rem_den + rem_num - 1) // rem_num)
        if d_start > 10000000:
            return

        if remaining_terms == 1:
            if rem_num == 1:
                d = rem_den
                if d >= d_start and d <= 10000000 and d >= min_d:
                    current.append(d)
                    if len(current) < best_count:
                        best_count = len(current)
                        best_sol = current[:]
                    current.pop()
            return

        d = d_start
        while d <= 10000000:
            if time.time() > deadline:
                return

            new_rem_num = rem_num * d - rem_den
            new_rem_den = rem_den * d

            if new_rem_num <= 0:
                break

            if new_rem_num * (d + 1) > (remaining_terms - 1) * new_rem_den:
                break

            current.append(d)
            dfs(new_rem_num, new_rem_den, d + 1, current, remaining_terms - 1)
            current.pop()

            d += 1

    for target_count in range(8, min(best_count, 12)):
        if time.time() > deadline:
            break

        dfs(1, 1, 5, [], target_count)

        if best_count <= target_count:
            break

    return best_sol
