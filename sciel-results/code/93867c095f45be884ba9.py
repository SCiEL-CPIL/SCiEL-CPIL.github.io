"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import factorial
    from functools import lru_cache

    deadline = time.time() + 85.0  # safety margin before 90s kill

    # 12! = 479001600
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target = factorial(12)

    # Generate all divisors of 12! that are > 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if val > 12:
                            divisors.append(val)

    divisors.sort()

    # Precompute exponent tuples for each divisor
    divisor_info = []
    for val in divisors:
        e2 = 0
        temp = val
        while temp % 2 == 0:
            e2 += 1
            temp //= 2
        e3 = 0
        while temp % 3 == 0:
            e3 += 1
            temp //= 3
        e5 = 0
        while temp % 5 == 0:
            e5 += 1
            temp //= 5
        e7 = 0
        while temp % 7 == 0:
            e7 += 1
            temp //= 7
        e11 = 0
        while temp % 11 == 0:
            e11 += 1
            temp //= 11
        divisor_info.append((val, (e2, e3, e5, e7, e11)))

    # Target exponents
    target_exp = (10, 5, 2, 1, 1)

    # Binary search on the maximum factor M
    # For a given M, check if we can decompose 12! into distinct factors in (12, M]

    def is_feasible(M):
        # Filter divisors <= M
        candidates = [(val, exp) for val, exp in divisor_info if val <= M]
        if not candidates:
            return None

        # Memoized DFS: state = (remaining_exponents, start_index)
        # Returns a list of values or None
        memo = {}

        def dfs(rem_exp, start_idx):
            if rem_exp == (0, 0, 0, 0, 0):
                return []

            key = (rem_exp, start_idx)
            if key in memo:
                return memo[key]

            if time.time() > deadline:
                return None

            # Try each candidate from start_idx onwards
            for i in range(start_idx, len(candidates)):
                val, exp = candidates[i]
                # Check if we can subtract these exponents
                if all(exp[j] <= rem_exp[j] for j in range(5)):
                    new_rem = tuple(rem_exp[j] - exp[j] for j in range(5))
                    result = dfs(new_rem, i + 1)
                    if result is not None:
                        memo[key] = [val] + result
                        return [val] + result

            memo[key] = None
            return None

        return dfs(target_exp, 0)

    # Binary search for the minimum M
    low = 13
    high = target
    best_factors = None

    while low <= high:
        mid = (low + high) // 2
        factors = is_feasible(mid)
        if factors is not None:
            best_factors = factors
            high = mid - 1
        else:
            low = mid + 1

    if best_factors is None:
        # Fallback: try the maximum M one more time
        best_factors = is_feasible(target)

    if best_factors is None:
        # Ultimate fallback: return the whole 12! as a single factor
        best_factors = [target]

    return best_factors

# EVOLVE-BLOCK-END
