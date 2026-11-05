import time
from math import factorial, isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11

    best = [target]
    best_max = target

    # Precompute all divisors of 12!
    all_divisors = []
    for i in range(1, isqrt(target) + 1):
        if target % i == 0:
            d1 = i
            d2 = target // i
            if d1 > 12:
                all_divisors.append(d1)
            if d2 != d1 and d2 > 12:
                all_divisors.append(d2)
    all_divisors = sorted(set(all_divisors))

    def search(remaining, min_factor, current_factors, max_so_far, used_set):
        nonlocal best, best_max, start_time

        if time.time() - start_time > DEADLINE:
            return

        if remaining == 1:
            if len(current_factors) >= 1:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return

        if len(current_factors) >= 12:
            return

        if max_so_far >= best_max:
            return

        min_possible = max(min_factor, 13)

        if remaining < min_possible:
            return

        # Upper bound for this factor: must be < best_max and <= remaining
        hi = min(best_max - 1, remaining)

        if hi < min_possible:
            return

        # Get valid divisors from precomputed list
        divs = [d for d in all_divisors if min_possible <= d <= hi and d not in used_set]

        if not divs:
            return

        # Sort: prefer divisors close to geometric mean for balanced splits
        max_factors_left = 12 - len(current_factors)
        if max_factors_left > 1:
            target_size = remaining ** (1.0 / max_factors_left)
            divs.sort(key=lambda d: abs(d - target_size))
        else:
            divs.sort()

        # Limit to speed up
        max_divs = 100
        divs = divs[:max_divs]

        for d in divs:
            new_max = max(max_so_far, d)
            if new_max >= best_max:
                continue

            new_used = used_set | {d}
            new_factors = current_factors + [d]
            new_min = d + 1

            search(remaining // d, new_min, new_factors, new_max, new_used)

            if time.time() - start_time > DEADLINE:
                return

    search(target, 13, [], 0, set())

    return best
