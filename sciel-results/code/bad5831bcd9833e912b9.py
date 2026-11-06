"""Rewritten: exact minimum-term divisor-sum via 0/1 knapsack DP over divisors of 5040."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 60.0  # safety margin under 90s kill
    n = 5040
    target = 4979

    # Enumerate all positive divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    # Sort descending to prefer larger divisors in ties (helps reconstruction determinism)
    divisors.sort(reverse=True)

    INF = float('inf')
    # dp[s] = minimum number of divisors to sum to exactly s
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent tracking: for each (item_index, sum) we record whether item was used
    # We'll store prev pointer: used[item][s] = True if item included to reach s with optimal count
    used = [[False] * (target + 1) for _ in range(len(divisors))]

    for i, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # 0/1 knapsack: iterate s from target down to d
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                used[i][s] = True

    if dp[target] == INF:
        # Fallback: greedy (should not happen since sum of all divisors > target)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0
        return chosen

    # Reconstruct
    chosen = []
    s = target
    for i in range(len(divisors) - 1, -1, -1):
        if used[i][s]:
            chosen.append(divisors[i])
            s -= divisors[i]
    assert s == 0
    assert sum(chosen) == target
    # Ensure distinctness
    assert len(set(chosen)) == len(chosen)
    # Ensure each is a divisor of 5040
    assert all(n % d == 0 for d in chosen)
    return chosen
# EVOLVE-BLOCK-END
