"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    deadline = t0 + 80

    n = 5040
    target = 4979

    # Generate all divisors of n
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort(reverse=True)
    m = len(divisors)

    # 1. Dynamic Programming for minimum cardinality subset sum
    INF = float('inf')
    max_sum = target
    dp = [INF] * (max_sum + 1)
    dp[0] = 0
    prev_sum = [-1] * (max_sum + 1)
    prev_div_idx = [-1] * (max_sum + 1)

    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Process in reverse to avoid using the same divisor twice
        for s in range(max_sum, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_div_idx[s] = idx

    if dp[target] != INF:
        # Reconstruct the solution
        chosen = []
        s = target
        while s > 0:
            idx = prev_div_idx[s]
            if idx < 0:
                break
            chosen.append(divisors[idx])
            s = prev_sum[s]

        # Verify the solution
        if len(set(chosen)) == len(chosen) and sum(chosen) == target and all(n % d == 0 for d in chosen):
            return chosen

    # Fallback: Iterative Deepening DFS with pruning
    # Precompute suffix sums for max possible sum with k items from index i onwards
    # sorted in descending order
    suffix_max = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + divisors[i]

    def dfs(remaining, start_idx, chosen, depth):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return chosen[:]
        if len(chosen) >= depth:
            return None

        # Pruning: even taking all remaining items can't reach target
        if suffix_max[start_idx] < remaining:
            return None

        slots = depth - len(chosen)

        # Pruning: not enough slots
        if m - start_idx < slots:
            return None

        # Pruning: max sum with slots items from start_idx is insufficient
        if suffix_max[start_idx + slots] < remaining:
            return None

        for i in range(start_idx, m):
            d = divisors[i]
            if d > remaining:
                continue
            chosen.append(d)
            result = dfs(remaining - d, i + 1, chosen, depth)
            if result is not None:
                return result
            chosen.pop()
        return None

    for depth in range(1, 15):
        result = dfs(target, 0, [], depth)
        if result is not None:
            return result

    # Last resort: Greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen

# EVOLVE-BLOCK-END
