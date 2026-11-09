"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040 efficiently
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    # Filter: only divisors <= target can be useful
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    INF = float('inf')
    
    # dp[s] = minimum number of divisors to achieve sum s
    # Use numpy for speed
    dp = np.full(target + 1, INF, dtype=np.float64)
    dp[0] = 0.0
    
    # parent[s] = (prev_sum, divisor_index) for reconstruction
    parent_prev = np.full(target + 1, -1, dtype=np.int32)
    parent_idx = np.full(target + 1, -1, dtype=np.int32)
    
    # Process each divisor (0/1 knapsack style)
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        
        # Update dp in reverse to avoid using same divisor twice
        # Vectorized: for s from target down to d
        # dp[s] = min(dp[s], dp[s-d] + 1)
        
        # We need to do this carefully to avoid race conditions
        # Process in chunks or use a temporary array
        # Since we're going in reverse, we can use numpy slicing
        
        # Get the slice dp[d:target+1] and dp[0:target+1-d]
        # New values: dp[d:] = min(dp[d:], dp[:-d] + 1)
        
        # But we need to track parents only when improvement happens
        # So we do it in a loop but with numpy for the comparison
        
        s_range = np.arange(target, d - 1, -1)
        prev_s = s_range - d
        
        # Current dp values at s_range
        current_dp = dp[s_range]
        prev_dp = dp[prev_s]
        
        # New candidate values
        new_vals = prev_dp + 1.0
        
        # Where improvement happens
        improved = new_vals < current_dp
        
        if np.any(improved):
            # Update dp
            dp[s_range[improved]] = new_vals[improved]
            
            # Update parent tracking (need to do this per-element for correctness)
            # Since this is rare after initial iterations, loop is fine
            for i in np.where(improved)[0]:
                s = s_range[i]
                parent_prev[s] = prev_s[i]
                parent_idx[s] = idx
    
    # Check if target is reachable
    if dp[target] == INF:
        # Fallback: greedy (largest first)
        remainder = target
        chosen = []
        for d in reversed(divisors):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s = parent_prev[s]
        div_idx = parent_idx[s]
        if div_idx < 0:
            # Shouldn't happen if dp[target] is finite
            break
        chosen.append(divisors[div_idx])
        s = prev_s
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors not distinct"
    assert all(n % d == 0 for d in chosen), "Not all are divisors of 5040"
    assert sum(chosen) == target, f"Sum is {sum(chosen)}, expected {target}"
    assert len(chosen) <= 256, f"Too many divisors: {len(chosen)}"
    
    return chosen

# EVOLVE-BLOCK-END
