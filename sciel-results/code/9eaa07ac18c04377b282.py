"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0

    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()

    m = len(divisors)  # should be 60

    # DP approach: 0/1 knapsack minimizing count
    # dp[s] = (min_count, list_of_indices) -- but storing lists is expensive
    # Instead: dp[s] = min count, and pred[s] = (prev_sum, divisor_index) for reconstruction
    # Process divisors one by one (0/1 knapsack)

    # We need at most 'target' sums, and at most m terms
    # dp[s] = minimum number of divisors to achieve sum s, or -1 if unreachable
    # Use array of size target+1
    INF = m + 1  # any value > m is "unreachable"
    dp = [INF] * (target + 1)
    dp[0] = 0

    # pred[s] stores (previous_sum, divisor_idx) to reconstruct the solution
    pred = [None] * (target + 1)  # None means not yet set

    # Process each divisor (0/1 knapsack: iterate sums in descending order)
    for idx in range(m):
        d = divisors[idx]
        if d > target:
            continue
        # Update dp in descending order to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                pred[s] = (s - d, idx)

    if time.time() > DEADLINE:
        # Should not happen, but safety
        pass

    # Reconstruct the solution
    if dp[target] == INF:
        # Fallback: greedy (should not happen for this instance)
        remainder = target
        chosen = []
        for d in reversed(divisors):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        best_result = chosen
    else:
        # Reconstruct using pred
        chosen_indices = []
        s = target
        while s > 0:
            prev_s, idx = pred[s]
            chosen_indices.append(idx)
            s = prev_s
        best_result = [divisors[i] for i in chosen_indices]

    # Verify
    assert sum(best_result) == target, f"Sum mismatch: {sum(best_result)} != {target}"
    assert len(set(best_result)) == len(best_result), "Duplicate divisors found"
    for d in best_result:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    assert len(best_result) <= 256, f"Too many divisors: {len(best_result)}"

    return best_result
# EVOLVE-BLOCK-END
