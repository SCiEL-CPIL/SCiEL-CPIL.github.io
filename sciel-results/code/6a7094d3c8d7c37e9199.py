"""Exact minimum-cardinality subset-sum over divisors of 5040."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin under 90s wall-clock
    n = 5040
    target = 4979

    # All distinct positive divisors of 5040
    divs = [d for d in range(1, n + 1) if n % d == 0]
    m = len(divs)  # 60

    # DP: dp[s] = minimum number of divisors to reach sum s (inf if unreachable)
    # parent[s] = index of the divisor last used to reach s (for reconstruction)
    INF = 10**9
    dp = [INF] * (target + 1)
    parent = [-1] * (target + 1)
    dp[0] = 0

    # 0-1 knapsack style: iterate divisors, update sums downward to enforce distinctness
    for i, d in enumerate(divs):
        if d > target:
            continue
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = i

    if time.time() > deadline:
        # Should never happen for this tiny instance; fall back to greedy
        rem = target
        chosen = []
        for d in sorted(divs, reverse=True):
            if d <= rem:
                chosen.append(d)
                rem -= d
        return chosen

    # Reconstruct the optimal set
    chosen = []
    s = target
    while s > 0 and parent[s] != -1:
        i = parent[s]
        chosen.append(divs[i])
        s -= divs[i]
        if time.time() > deadline:
            break

    # Exact membership and sum checks
    assert len(set(chosen)) == len(chosen), "divisors must be distinct"
    assert all(n % x == 0 and x > 0 for x in chosen), "each must be a positive divisor of 5040"
    assert sum(chosen) == target, "sum must equal target"
    assert len(chosen) <= 256, "at most 256 divisors"

    return chosen
# EVOLVE-BLOCK-END
