"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _compute_solution():
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # DP: dp[s] = minimum number of divisors to achieve sum s
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_index) for reconstruction
    parent = [(-1, -1)] * (target + 1)
    
    # Process each divisor (0/1 knapsack style)
    # Sort divisors in descending order
    sorted_divs = sorted(divisors, reverse=True)
    
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    for idx, d in enumerate(sorted_divs):
        if time.time() > deadline:
            break
        # Process in reverse to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, idx)
    
    if dp[target] == INF:
        # Fallback to greedy if DP fails (shouldn't happen)
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, f"Could not reach target, remainder={remainder}"
        return chosen
    
    # Reconstruct the solution
    chosen_indices = []
    s = target
    while s > 0:
        prev_s, div_idx = parent[s]
        chosen_indices.append(div_idx)
        s = prev_s
    
    # Convert indices back to divisor values
    chosen = [sorted_divs[i] for i in chosen_indices]
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors not distinct"
    assert all(n % d == 0 for d in chosen), "Not all are divisors of 5040"
    assert sum(chosen) == target, f"Sum is {sum(chosen)}, expected {target}"
    assert len(chosen) <= 256, f"Too many divisors: {len(chosen)}"
    
    return chosen

# Pre-compute the solution at module load time
_PRECOMPUTED = _compute_solution()

def construct():
    return _PRECOMPUTED

# EVOLVE-BLOCK-END
