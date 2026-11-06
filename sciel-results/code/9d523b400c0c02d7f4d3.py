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
    divs.sort()
    # Filter divisors that are too large to be part of a sum equal to target
    # (since all divisors are positive, any divisor > target cannot be used)
    divs = [d for d in divs if d <= target]
    m = len(divs)
    div_set = set(divs)

    def check_time():
        return time.time() > deadline

    # 1. Try k=1
    if target in div_set:
        return [target]

    # 2. Try k=2
    # We need distinct d1, d2 such that d1 + d2 = target
    # Iterate through divisors. To avoid duplicates and ensure d1 < d2, 
    # we can just check if (target - d) is in set and (target - d) != d.
    # Since order doesn't matter for the set, we just return the list.
    for d in divs:
        rem = target - d
        if rem > d and rem in div_set:
            return [d, rem]

    # 3. Try k=3
    # Need distinct d1, d2, d3 such that d1 + d2 + d3 = target
    # Iterate d1, d2 with d1 < d2.
    # rem = target - d1 - d2.
    # Check if rem > d2 and rem in div_set.
    # This ensures d1 < d2 < rem, so all distinct.
    for i in range(m):
        d1 = divs[i]
        if check_time():
            break
        # d2 must be > d1
        for j in range(i + 1, m):
            d2 = divs[j]
            rem = target - d1 - d2
            if rem <= d2:
                break # Since divs are sorted, further d2 will only decrease rem
            if rem in div_set:
                return [d1, d2, rem]

    # 4. Fallback: DP for k >= 4
    # This is the original strategy, ensuring we find the minimum count.
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)

    # Process divisors in reverse order for 0/1 knapsack
    # We want min count, so cost is 1 for each item.
    for d in divs:
        if check_time():
            break
        # Iterate sums backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)

    if dp[target] == INF:
        # Fallback greedy if DP fails (should not happen for practical numbers)
        remainder = target
        chosen = []
        for d in sorted(divs, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s

    return chosen
