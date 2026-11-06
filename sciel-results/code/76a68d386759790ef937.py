import time

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s kill
    n = 5040
    target = 4979

    # Generate divisors of 5040
    divs = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    divs.sort(reverse=True)  # largest first
    # Filter: only divisors <= target can be used
    divs = [d for d in divs if d <= target]

    # 0/1 knapsack DP: dp[s] = minimum number of distinct divisors to achieve sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent[s] = (prev_sum, divisor) for reconstruction
    parent = [None] * (target + 1)

    for d in divs:
        if time.time() > deadline:
            break
        # 0/1 knapsack: iterate backwards to avoid reusing same divisor
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)

    if dp[target] == INF:
        # Fallback: greedy (should not happen for this instance)
        remainder = target
        chosen = []
        for d in divs:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        if remainder == 0:
            return chosen
        return chosen

    # Reconstruct the optimal solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s

    return chosen
