"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    divisors.sort()
    
    num_divs = len(divisors)
    
    # dp[s] = (min_count, bitmask) where bitmask represents which divisors are used
    # Initialize with infinity (large number)
    INF = float('inf')
    dp = [(INF, 0) for _ in range(target + 1)]
    dp[0] = (0, 0)
    
    # Process each divisor
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Update dp in reverse order to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d][0] + 1 < dp[s][0]:
                dp[s] = (dp[s - d][0] + 1, dp[s - d][1] | (1 << idx))
    
    # Extract the solution
    if dp[target][0] == INF:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in reversed(divisors):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Reconstruct which divisors were used
    bitmask = dp[target][1]
    result = []
    for idx in range(num_divs):
        if bitmask & (1 << idx):
            result.append(divisors[idx])
    
    return result

# EVOLVE-BLOCK-END
