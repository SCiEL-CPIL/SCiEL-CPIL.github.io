import time

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 50  # 50s safety margin

    # Generate divisors efficiently
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    
    # Filter divisors that are too large
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)

    # DP: dp[s] = min count to reach sum s
    INF = 10**9
    dp = [INF] * (target + 1)
    prev = [(-1, -1)] * (target + 1)
    dp[0] = 0

    for d in divisors:
        if time.time() > deadline:
            break
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev[s] = (s - d, d)

    if dp[target] == INF:
        return []

    # Reconstruct
    res = []
    s = target
    while s > 0:
        ps, d = prev[s]
        if d == -1:
            return [] # Safety break if reconstruction fails
        res.append(d)
        s = ps

    return res
