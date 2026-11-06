import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    n = 5040
    target = 4979

    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors to ensure consistent ordering for DP
    divisors.sort()
    num_divs = len(divisors)
    
    # Filter divisors > target as they can never be part of a sum equal to target
    # (Since all divisors are positive)
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    INF = float('inf')
    
    # dp[i][s] = minimum number of divisors from the first i divisors 
    # that sum to s.
    # Dimensions: (num_divs + 1) x (target + 1)
    # We initialize with INF.
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    
    # Base case: sum 0 with 0 divisors is 0
    for i in range(num_divs + 1):
        dp[i][0] = 0
        
    # Fill the DP table
    # i goes from 1 to num_divs
    # divisors[i-1] is the current divisor being considered
    for i in range(1, num_divs + 1):
        d = divisors[i - 1]
        # If time is running out, we break, but this should be very fast
        if time.time() > deadline:
            break
            
        # s goes from 0 to target
        for s in range(target + 1):
            # Option 1: Don't take divisor d
            val_not_take = dp[i - 1][s]
            
            # Option 2: Take divisor d (if s >= d)
            if s >= d:
                prev_count = dp[i - 1][s - d]
                if prev_count != INF:
                    val_take = prev_count + 1
                    if val_take < val_not_take:
                        val_not_take = val_take
            
            dp[i][s] = val_not_take

    # Check if a solution exists
    min_count = dp[num_divs][target]
    
    if min_count == INF:
        # Fallback: This shouldn't happen for practical numbers like 5040
        # but we need to return something valid.
        # Greedy fallback: pick largest divisors first
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        return chosen

    # Backtrack to find the specific divisors
    selected = []
    s = target
    i = num_divs
    
    while i > 0 and s > 0:
        # Check if the current divisor divisors[i-1] was used
        d = divisors[i - 1]
        
        # If dp[i][s] == dp[i-1][s], then we didn't take d
        # If dp[i][s] == dp[i-1][s-d] + 1, then we took d
        if s >= d and dp[i][s] == dp[i - 1][s - d] + 1:
            selected.append(d)
            s -= d
            i -= 1
        else:
            i -= 1
            
    return selected
