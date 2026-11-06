"""Rewritten program using exact DP for minimum-cardinality subset sum."""
import time


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    num_divs = len(divisors)

    # DP: dp[s] = (min_count, prev_sum, divisor_index_used)
    # We want minimum number of divisors summing to target.
    # Use a dict or array. target is 4979, manageable.
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_prev = [-1] * (target + 1)   # previous sum
    dp_div = [-1] * (target + 1)    # index of divisor used to reach this state
    dp_count[0] = 0

    # Iterate divisors in some order; to find min count we process all.
    # Standard 0/1 knapsack style: for each divisor, update sums in reverse.
    # But we want min count, so for each divisor d, for s from target down to d:
    #   if dp_count[s-d] + 1 < dp_count[s]: update
    for idx, d in enumerate(divisors):
        if d > target:
            continue
        # reverse to avoid reusing same divisor
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_prev[s] = s - d
                dp_div[s] = idx

    if dp_count[target] == INF:
        # Fallback: should not happen for this instance, but provide greedy
        rem = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= rem:
                chosen.append(d)
                rem -= d
            if rem == 0:
                break
        if rem == 0:
            return chosen
        return []

    # Reconstruct the set of divisors
    result = []
    s = target
    while s > 0:
        idx = dp_div[s]
        result.append(divisors[idx])
        s = dp_prev[s]

    result.sort()
    # Verify
    if sum(result) == target and len(set(result)) == len(result):
        return result
    # Fallback
    return []


# EVOLVE-BLOCK-END
