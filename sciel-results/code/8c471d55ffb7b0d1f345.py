import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
        
    # Filter divisors <= target
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)
    
    # Check if target is a divisor
    if target in divisors:
        return [target]
        
    # DP using NumPy for speed
    # dp[s] = min number of divisors to sum to s
    # We use a large number as INF
    INF = 1000
    dp = np.full(target + 1, INF, dtype=np.int32)
    dp[0] = 0
    
    # parent tracking for reconstruction
    # parent[s] = (prev_sum, divisor)
    # Using lists of tuples is easier for reconstruction than numpy arrays of tuples
    parent = [None] * (target + 1)
    
    for d in divisors:
        if time.time() > DEADLINE:
            break
            
        # Update dp in reverse order (0/1 knapsack)
        # s from target down to d
        # dp[s] = min(dp[s], dp[s-d] + 1)
        
        # Vectorized update is tricky with the "reverse" dependency in 0/1 knapsack 
        # because dp[s] depends on dp[s-d] which might have been updated in the same iteration 
        # if we go forward. Going backward avoids this.
        # However, NumPy doesn't support "in-place" backward updates easily without loops or complex slicing.
        # Given the small size (60 * 5000), a pure Python loop is likely faster than 
        # setting up NumPy overhead and doing slow indexed updates or complex slicing.
        
        # Let's stick to pure Python for the DP loop as it is very fast for this size,
        # but optimize the inner loop with local variables.
        
        # Actually, let's try to use a list for dp to allow fast indexing.
        # dp_list = dp.tolist()
        
        pass

    # Reverting to pure Python DP as it is optimal and fast enough, 
    # but adding better local variable caching.
    
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)
    
    for d in divisors:
        if time.time() > DEADLINE:
            break
        # Local references for speed
        dp_local = dp
        par_local = parent
        for s in range(target, d - 1, -1):
            if dp_local[s - d] + 1 < dp_local[s]:
                dp_local[s] = dp_local[s - d] + 1
                par_local[s] = (s - d, d)
                
    if dp[target] == INF:
        # Fallback greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
        
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
        
    return chosen
