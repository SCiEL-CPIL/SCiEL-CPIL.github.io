import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    n = 5040
    target = 4979

    # Get all positive divisors of n
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    # Sort in descending order to prioritize larger divisors (fewer terms)
    divisors.sort(reverse=True)

    num_divs = len(divisors)

    INF = float('inf')

    # 1D DP with parent pointers for reconstruction
    # dp[s] = min number of divisors to sum to s
    # parent[s] = index of divisor in divisors[] that was last added to reach s
    # prev_sum[s] = the sum before adding parent[s], i.e., s - divisors[parent[s]]
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [-1] * (target + 1)
    prev_sum = [0] * (target + 1)

    for i in range(num_divs):
        d = divisors[i]
        if time.time() > deadline:
            break
        # Iterate sums in descending order to ensure 0/1 property (each divisor used at most once)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = i
                prev_sum[s] = s - d

    min_count = dp[target]

    if min_count == INF:
        # Fallback: try branch-and-bound
        return _branch_and_bound(divisors, target, n, start_time, deadline)

    # Reconstruct the solution by following parent pointers
    result = []
    s = target
    while s > 0:
        idx = parent[s]
        if idx == -1:
            break
        result.append(divisors[idx])
        s = prev_sum[s]

    # Verify the solution
    if sum(result) != target:
        return _branch_and_bound(divisors, target, n, start_time, deadline)

    # Ensure distinctness (should be guaranteed by 0/1 DP, but verify)
    if len(set(result)) != len(result):
        return _branch_and_bound(divisors, target, n, start_time, deadline)

    # Ensure all are divisors of n
    for d in result:
        if n % d != 0:
            return _branch_and_bound(divisors, target, n, start_time, deadline)

    return sorted(result, reverse=True)


def _branch_and_bound(divisors, target, n, start_time, deadline):
    """Branch-and-bound fallback: DFS with greedy upper bound and feasibility pruning."""
    # Greedy upper bound: take largest divisors until we can reach target
    def greedy_bound(divs, tgt):
        count = 0
        remaining = tgt
        for d in divs:
            if remaining <= 0:
                break
            if d <= remaining:
                remaining -= d
                count += 1
        return count if remaining == 0 else float('inf')

    best_count = greedy_bound(divisors, target)
    if best_count == float('inf'):
        return []

    best_result = None

    # Sort descending for better pruning
    divs = divisors[:]

    # Compute suffix sums for feasibility check: can we still reach target from remaining divisors?
    suffix_sum = [0] * (len(divs) + 1)
    for i in range(len(divs) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divs[i]

    def dfs(idx, current_sum, count, chosen):
        nonlocal best_count, best_result

        if time.time() > deadline:
            return

        if current_sum == target:
            if count < best_count:
                best_count = count
                best_result = chosen[:]
            return

        if current_sum > target:
            return

        if idx >= len(divs):
            return

        # Pruning: even taking all remaining divisors can't reach target
        if current_sum + suffix_sum[idx] < target:
            return

        # Pruning: minimum additional terms needed
        remaining_needed = target - current_sum
        if remaining_needed > 0:
            # Lower bound: we need at least ceil(remaining_needed / max_available) more terms
            # But more simply: if count + 1 >= best_count, no point continuing
            if count + 1 >= best_count:
                return

        # Option 1: skip divs[idx]
        dfs(idx + 1, current_sum, count, chosen)

        # Option 2: take divs[idx]
        if divs[idx] <= target - current_sum:
            chosen.append(divs[idx])
            dfs(idx + 1, current_sum + divs[idx], count + 1, chosen)
            chosen.pop()

    dfs(0, 0, 0, [])

    if best_result is not None:
        return sorted(best_result, reverse=True)
    return []
