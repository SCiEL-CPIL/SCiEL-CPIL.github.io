"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Generate all divisors of 5040
    n = 5040
    target = 4979
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors in descending order for better pruning
    divisors.sort(reverse=True)
    num_div = len(divisors)
    
    # DP: dp[s] = (min_count, tuple_of_indices) to achieve sum s
    # We only need sums up to target
    # Use a dict for sparse representation
    INF = float('inf')
    
    # dp[s] stores the minimum number of divisors to reach sum s
    # and we store parent pointers for reconstruction
    # Since target is 4979, we can use an array
    dp_count = [INF] * (target + 1)
    dp_parent = [-1] * (target + 1)  # which divisor index was used
    dp_prev_sum = [-1] * (target + 1)  # previous sum before adding this divisor
    
    dp_count[0] = 0
    
    # Process each divisor (0/1 knapsack style)
    # For minimum cardinality, we process each divisor once
    # We need to be careful: process in a way that each divisor is used at most once
    # Standard 0/1 knapsack: iterate sums in reverse for each item
    
    for i, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Process in reverse to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_parent[s] = i
                dp_prev_sum[s] = s - d
    
    # Check if we can reach target
    if dp_count[target] == INF:
        # Fallback to greedy (shouldn't happen, but safety)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, f"Could not reach target, remainder={remainder}"
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        idx = dp_parent[s]
        chosen.append(divisors[idx])
        s = dp_prev_sum[s]
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors not distinct"
    assert all(n % d == 0 for d in chosen), "Not all are divisors of 5040"
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    assert len(chosen) <= 256, "Too many divisors"
    
    return chosen
# EVOLVE-BLOCK-END
