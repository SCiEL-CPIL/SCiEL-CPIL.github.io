"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # process larger divisors first for pruning
    
    num_divs = len(divisors)
    
    # dp[s] = (min_count, bitmask_of_divisors_used)
    # Use a large number as infinity
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_mask = [0] * (target + 1)
    dp_count[0] = 0
    dp_mask[0] = 0
    
    # For each divisor, update dp in reverse to avoid using same divisor twice
    for idx, d in enumerate(divisors):
        if d > target:
            continue
        bit = 1 << idx
        # Process sums in reverse to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_mask[s] = dp_mask[s - d] | bit
    
    if dp_count[target] == INF:
        # Fallback: should not happen for practical numbers, but handle gracefully
        # Try greedy as fallback
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Reconstruct the solution
    mask = dp_mask[target]
    chosen = []
    for idx, d in enumerate(divisors):
        if mask & (1 << idx):
            chosen.append(d)
    
    assert sum(chosen) == target, f"Sum check failed: {sum(chosen)} != {target}"
    assert len(set(chosen)) == len(chosen), "Duplicates found"
    for d in chosen:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return chosen
# EVOLVE-BLOCK-END
