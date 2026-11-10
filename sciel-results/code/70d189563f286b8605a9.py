"""Rewritten program: search for a partition of 12! into distinct integers > 12, minimizing the largest factor."""
import time
from math import factorial
from functools import lru_cache
import bisect

TARGET = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11

# Precompute all divisors of TARGET that are > 12
def _get_all_divisors():
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    # Keep only those > 12
    divisors = sorted([d for d in divisors if d > 12])
    return divisors

_ALL_DIVS = _get_all_divisors()

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    # Fallback: single factor
    best_result = [TARGET]
    best_max = TARGET

    # We want to minimize the largest factor.
    # Strategy: iterate candidate "max factor" bounds in increasing order.
    # For each bound B (a divisor of TARGET > 12), check if there exists a subset
    # of distinct divisors from _ALL_DIVS (each <= B, > 12) whose product is TARGET.
    # The first (smallest) B for which this is true gives the optimal answer.

    # To make this efficient, we use a DFS with memoization.
    # Key insight: we can restrict the search to divisors <= B.
    # Also, we can prune: if the product of the remaining target cannot be formed
    # by the remaining factors, stop.

    # Let's iterate over candidate bounds. Since _ALL_DIVS is sorted, for bound = _ALL_DIVS[k],
    # the valid divisors are _ALL_DIVS[0:k+1].

    n_divs = len(_ALL_DIVS)

    # We'll try bounds in increasing order. For each, run a DFS to find a partition.
    # The DFS will try to pick factors from the available divisors (<= bound) such that
    # their product equals TARGET, using distinct factors.

    # Optimization: we can precompute for each bound the list of divisors <= bound.
    # But building this each time is O(n). Instead, we'll pass the full list and a limit index.

    # Let's use a memoized DFS. The state is (start_idx, target, remaining_factors_count).
    # But we don't fix the number of factors; we just need ANY valid partition.
    # Actually, the number of factors is variable (1..12). So we need to search over all possibilities.

    # A better approach: DFS that picks factors one by one, ensuring distinctness and product = TARGET.
    # We'll iterate over the divisors in increasing order (up to the bound) and try to include or exclude each.
    # But that's exponential. Instead, we use a recursive search with pruning.

    # Let's define a recursive function:
    # dfs(idx, tgt, used) -> returns a list of factors or None
    # where idx is the index in _ALL_DIVS (only consider divisors <= bound),
    # tgt is the remaining target,
    # used is a set of indices already used (to ensure distinctness).
    # But used as a set is expensive for memoization. Instead, we can use a bitmask or just pass the start index.

    # Actually, since we're picking distinct factors and the divisors are sorted, we can use:
    # dfs(start_idx, tgt) -> list of factors or None
    # where we only consider divisors from start_idx onward (to ensure distinctness and avoid re-picking).

    # But we also need to ensure all factors are <= bound. So we'll limit the search to divisors[0:limit_idx+1].

    # Let's try this approach. We'll iterate over bounds, and for each bound, run the DFS.

    # To speed up, we'll use memoization with (start_idx, tgt) as the key.
    # But tgt can be large, and start_idx up to n_divs. The state space might be large.
    # However, tgt is always a divisor of TARGET, so there are at most ~1000 distinct tgt values.
    # And start_idx is up to ~1000. So the state space is ~1,000,000, which is manageable.

    # Let's implement this.

    # We'll try bounds in increasing order. For each bound, we set limit_idx = k (where bound = _ALL_DIVS[k]).
    # Then we run dfs(0, TARGET) with the constraint that we only pick from _ALL_DIVS[0:limit_idx+1].

    # But we need to reset the memo for each bound? No, the memo depends on (start_idx, tgt), and the available
    # divisors are _ALL_DIVS[0:limit_idx+1]. So the memo is valid for a given limit_idx.
    # But as we increase limit_idx, the available divisors change. So we need a separate memo for each bound.
    # That's expensive. Instead, we can just run the DFS without memoization across bounds, and use memoization within a single bound search.

    # Let's try: for each bound, create a new memo and run the DFS.

    # To limit the work, we'll check time periodically.

    for k in range(n_divs):
        if time.time() > deadline - 5.0:
            break

        bound = _ALL_DIVS[k]
        # The available divisors are _ALL_DIVS[0:k+1]
        # We need to find a subset of these (distinct) whose product is TARGET.

        # Quick prune: if the product of all available divisors is less than TARGET, skip.
        # But that's expensive to compute. Instead, note that if the largest available divisor is < TARGET,
        # we need multiple factors. Let's just run the DFS.

        # Define the DFS with memoization.
        memo = {}

        def dfs(start_idx, tgt):
            if tgt == 1:
                return []
            if start_idx > k:  # no more divisors to consider
                return None
            key = (start_idx, tgt)
            if key in memo:
                return memo[key]

            for i in range(start_idx, k + 1):
                d = _ALL_DIVS[i]
                if d > tgt:
                    break  # since divisors are sorted
                if tgt % d != 0:
                    continue
                sub = dfs(i + 1, tgt // d)
                if sub is not None:
                    memo[key] = [d] + sub
                    return [d] + sub
            memo[key] = None
            return None

        result = dfs(0, TARGET)
        if result is not None:
            # Found a valid partition with all factors <= bound
            actual_max = max(result)
            if actual_max < best_max:
                best_max = actual_max
                best_result = result
                # Since we're iterating bounds in increasing order, this is the best possible.
                # Any solution with max factor < bound would have been found at an earlier bound.
                return best_result

    # If we exhausted the time or bounds, return best found
    return best_result
