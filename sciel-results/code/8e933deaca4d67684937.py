import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill

    target = factorial(12)  # 479001600

    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We need to partition these prime factors into k distinct integers > 12,
    # minimizing the largest factor.

    # Strategy: DFS over factor assignments.
    # State: (remaining_target, last_factor, count, current_factors)
    # We build factors in increasing order to ensure distinctness.

    # Precompute all divisors of target for quick lookup
    all_divs = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    all_divs = sorted(set(all_divs))
    all_divs_set = set(all_divs)

    # Cache for divisors of a number that are > 12
    divs_cache = {}

    def get_divs_gt12(n):
        if n in divs_cache:
            return divs_cache[n]
        result = []
        limit = int(n**0.5) + 1
        for i in range(1, limit):
            if n % i == 0:
                if i > 12:
                    result.append(i)
                other = n // i
                if other != i and other > 12:
                    result.append(other)
        result.sort()
        divs_cache[n] = result
        return result

    best_max = target
    best_path = [target]

    def dfs(rem, last, count, path):
        nonlocal best_max, best_path, start_time

        if time.time() - start_time > deadline:
            return

        if rem == 1:
            if 1 <= count <= 12:
                current_max = path[-1]
                if current_max < best_max:
                    best_max = current_max
                    best_path = list(path)
            return

        if count == 12:
            if rem > last and rem > 12 and rem < best_max:
                best_max = rem
                best_path = path + [rem]
            return

        # Pruning: if rem is not a divisor of target, we can't proceed (shouldn't happen)
        # Get candidate divisors of rem that are > last and < best_max
        candidates = get_divs_gt12(rem)
        
        valid_candidates = []
        for d in candidates:
            if d > last and d < best_max:
                valid_candidates.append(d)
            elif d >= best_max:
                break  # Sorted, so no need to check further

        # Try candidates in ascending order to find smaller max values earlier
        for d in valid_candidates:
            # Additional pruning:
            # If we pick d, the remaining product is rem // d.
            # We need to form (12 - count - 1) more factors, each > d.
            # The smallest possible product for m factors all > d is (d+1)*(d+2)*...*(d+m)
            # But we don't have to use exactly 12 factors.
            # However, if rem // d is less than d+1, we can't pick any more factors > d,
            # so rem // d must be 1 (handled above) or we must be able to continue.
            
            # If rem // d < d + 1, then we can't pick any factor > d from rem // d.
            # So rem // d must be 1. If rem // d != 1, skip.
            if rem // d != 1 and rem // d < d + 1:
                continue

            dfs(rem // d, d, count + 1, path + [d])

            if time.time() - start_time > deadline:
                return

    dfs(target, 12, 0, [])

    return best_path
