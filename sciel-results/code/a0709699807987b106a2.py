"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate all distinct positive divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors in descending order for greedy-like heuristics
    divisors_desc = sorted(divisors, reverse=True)
    
    # Greedy solution first (the seed approach)
    remainder = target
    chosen_greedy = []
    for d in divisors_desc:
        if d <= remainder:
            chosen_greedy.append(d)
            remainder -= d
        if remainder == 0:
            break
    assert remainder == 0, f"Greedy failed, remainder={remainder}"
    best_solution = list(chosen_greedy)
    best_len = len(best_solution)

    if best_len == 1:
        return best_solution

    # DP: dp[s] = minimum number of divisors to sum to s
    # We use a dictionary or array. target is 4979, which is manageable.
    # dp[s] = min number of distinct divisors from our set that sum to s
    
    # Use array for speed
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # We'll track which divisors are used via parent pointers for reconstruction
    # But for minimum count, we can just track the count and then reconstruct.
    
    # Actually, for reconstruction we need to know which divisor was last added.
    # Let's use: dp[s] = (count, last_divisor) or just count and a separate parent array.
    
    # Better approach: dp[s] = min count, and parent[s] = (prev_sum, divisor_used)
    parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor)
    
    for d in divisors:  # iterate over each divisor once (0/1 knapsack style)
        # Go backwards to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] == INF:
        # Should not happen since greedy worked
        return best_solution
    
    if dp[target] >= best_len:
        return best_solution
    
    # Reconstruct the solution
    solution = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        solution.append(d)
        s = prev_s
    
    solution.reverse()
    
    if len(solution) < best_len:
        best_solution = solution
        best_len = len(solution)
    
    # Verify the solution
    assert len(set(best_solution)) == len(best_solution), "Not distinct"
    assert all(5040 % d == 0 for d in best_solution), "Not divisors"
    assert sum(best_solution) == target, f"Sum mismatch: {sum(best_solution)}"
    assert len(best_solution) <= 256, "Too many divisors"
    
    return best_solution
# EVOLVE-BLOCK-END
