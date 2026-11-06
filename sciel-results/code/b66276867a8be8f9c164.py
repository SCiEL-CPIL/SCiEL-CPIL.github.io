import time

def construct():
    start = time.time()
    deadline = start + 50.0
    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divs = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    divs.sort(reverse=True)
    divs = [d for d in divs if d <= target]
    m = len(divs)

    def check_time():
        return time.time() > deadline

    # DP: dp[s] = minimum number of divisors to achieve sum s
    # Use a large integer as infinity
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)

    for d in divs:
        if check_time():
            break
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)

    if dp[target] == INF:
        # Fallback greedy
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

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s

    return chosen
