import time

# EVOLVE-BLOCK-START
def construct():
    n = 5040
    target = 4979

    # Defensive deadline guard (the exact DP below runs in milliseconds).
    deadline = time.time() + 85.0

    # All positive divisors of 5040 (there are 60).
    divs = [d for d in range(1, n + 1) if n % d == 0]
    divs.sort()
    m = len(divs)

    # Exact 0/1 knapsack dynamic program minimizing the number of distinct
    # divisors whose sum equals `target`.  dp[i][s] = min count using the
    # first i divisors to reach sum s.
    INF = 10 ** 9
    dp = [[INF] * (target + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    choice = [[False] * (target + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        d = divs[i - 1]
        prev = dp[i - 1]
        cur = dp[i]
        cho = choice[i]
        for s in range(target + 1):
            best = prev[s]          # do not use divisor i
            take = False
            if s >= d:
                cand = prev[s - d] + 1
                if cand < best:
                    best = cand
                    take = True
            cur[s] = best
            cho[s] = take

    count = dp[m][target]

    # Fallback (should never trigger): greedy seed strategy.
    if count >= INF:
        remainder = target
        chosen = []
        for d in range(n, 0, -1):
            if n % d == 0 and d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen

    # Reconstruct an optimal set of distinct divisors.
    s = target
    i = m
    result = []
    while i > 0 and s > 0:
        if choice[i][s]:
            result.append(divs[i - 1])
            s -= divs[i - 1]
        i -= 1

    # Exact membership and sum checks.
    assert sum(result) == target
    assert len(set(result)) == len(result)
    for d in result:
        assert n % d == 0

    return result
# EVOLVE-BLOCK-END
