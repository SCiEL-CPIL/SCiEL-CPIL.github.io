import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill

    N = math.factorial(12)  # 479001600
    MIN_F = 13
    MAX_FACTORS = 12

    # Precompute all divisors of N
    def compute_divisors(n):
        divs = {1}
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                new_divs = set()
                power = 1
                for e in range(exp + 1):
                    for d in divs:
                        new_divs.add(d * power)
                    power *= p
                divs = new_divs
            p += 1 if p == 2 else 2
        if temp > 1:
            new_divs = set()
            for d in divs:
                new_divs.add(d)
                new_divs.add(d * temp)
            divs = new_divs
        return sorted(divs)

    all_divs = compute_divisors(N)
    valid_divs = [d for d in all_divs if d > MIN_F]
    valid_divs_set = set(valid_divs)
    n_valid = len(valid_divs)

    # Initial best: single factor N
    best_max = N
    best_factors = [N]

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # Binary search on the maximum factor M
    # For a given M, check if we can decompose N into k distinct factors, all in (12, M]
    # with product N.

    # Precompute which divisors are <= M for quick lookup
    # We'll do a DFS for feasibility given max factor M and number of factors k

    def can_decompose(remaining, min_idx, k, M, current_factors, time_limit):
        """Check if remaining can be factored into k distinct divisors > 12, each <= M,
        all greater than valid_divs[min_idx-1] (or > 12 if min_idx==0).
        Returns (success, factors_list) or (False, None)."""
        if time.time() > time_limit:
            return False, None

        if k == 0:
            if remaining == 1:
                return True, list(current_factors)
            return False, None

        if remaining < 1 or min_idx >= n_valid:
            return False, None

        if k == 1:
            if remaining <= M and remaining in valid_divs_set and (min_idx == 0 or remaining > valid_divs[min_idx - 1]):
                return True, current_factors + [remaining]
            return False, None

        # Try factors from min_idx upward, but only those <= M and dividing remaining
        for i in range(min_idx, n_valid):
            f = valid_divs[i]
            if f > M:
                break
            if f > remaining:
                break
            if remaining % f != 0:
                continue

            current_factors.append(f)
            success, factors = can_decompose(remaining // f, i + 1, k - 1, M, current_factors, time_limit)
            current_factors.pop()

            if success:
                return True, factors

        return False, None

    # Binary search on M
    # Lower bound: at least 13
    # Upper bound: N (single factor)
    # We want the smallest M such that there exists some k (1..12) with a valid decomposition

    # First, find the best (smallest M) by trying increasing M values
    # But M can be large. Let's binary search on M.

    # Actually, let's think about this differently.
    # The answer is likely small. Let's try to find it via a more targeted search.

    # Key insight: if we use k factors, the geometric mean is (N)^(1/k).
    # For k=4: (479001600)^(1/4) ≈ 148
    # For k=5: (479001600)^(1/5) ≈ 54
    # For k=6: (479001600)^(1/6) ≈ 28
    # For k=3: (479001600)^(1/3) ≈ 783

    # So for k=4, factors around 148. For k=5, around 54. For k=6, around 28.
    # The maximum factor will be at least the geometric mean.

    # Let's try to find solutions for small k first, since they tend to have smaller max.

    # For each k from 2 to 12, try to find a decomposition that minimizes the max factor.
    # Use binary search on M for each k.

    def find_best_for_k(k, time_limit):
        """Find the smallest M such that N can be decomposed into k distinct factors in (12, M]."""
        # Lower bound for M: at least ceil(N^(1/k)) but also > 12
        # Upper bound: N
        lo = 13
        hi = N

        # Quick check: is k=1 possible? M = N
        if k == 1:
            return N, [N]

        # Binary search on M
        best_M = None
        best_factors = None

        while lo <= hi and time.time() < time_limit:
            mid = (lo + hi) // 2

            # Check if feasible with max factor mid
            success, factors = can_decompose(N, 0, k, mid, [], time_limit)

            if success:
                best_M = mid
                best_factors = factors
                hi = mid - 1
            else:
                lo = mid + 1

        return best_M, best_factors

    # Try each k from 2 to 12
    # Start with k that likely gives smallest max
    # k=4: geometric mean ~148
    # k=5: geometric mean ~54
    # k=6: geometric mean ~28

    # Let's try k in order of likely optimality
    k_order = [4, 5, 6, 3, 7, 8, 2, 9, 10, 11, 12]

    for k in k_order:
        if time.time() > deadline:
            break

        # Theoretical lower bound on max factor for k factors
        # Each factor > 12, distinct, product = N
        # Minimum possible max is at least ceil(N^(1/k))
        # But also at least 13 + (k-1) for distinctness
        lower_bound = max(13 + k - 1, math.ceil(N ** (1.0 / k)))

        M, factors = find_best_for_k(k, deadline)

        if M is not None and M < best_max:
            best_max = M
            best_factors = factors

        # If we found a solution where max is close to the theoretical lower bound, we can stop
        if M is not None and M <= lower_bound + 10:
            break

    return best_factors
