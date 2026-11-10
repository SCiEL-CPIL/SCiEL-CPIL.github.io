import time
import sys

_N = 5040
_TARGET = 4979
_DEADLINE = 85.0

def _divisors_of(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def _greedy_solve(divisors, target):
    """Greedy: pick largest divisors first."""
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen if sum(chosen) == target else []

def construct():
    start = time.time()
    divisors = _divisors_of(_N)
    # Filter divisors that are too large to be useful
    divisors = [d for d in divisors if d <= _TARGET]
    
    # Fast path: try greedy first
    greedy_result = _greedy_solve(divisors, _TARGET)
    if greedy_result:
        return greedy_result
    
    # DP for minimum cardinality subset sum
    # dp[s] = minimum number of divisors to sum to s
    # parent[s] = the last divisor added to reach s with the current optimal count
    # The 1D DP with reverse iteration ensures 0/1 usage.
    
    INF = 10**9
    dp = [INF] * (_TARGET + 1)
    parent = [-1] * (_TARGET + 1)
    
    dp[0] = 0
    max_reachable = 0  # Track the maximum sum that has been reached
    
    # Sort divisors in descending order to potentially find solutions faster
    # and to allow early termination if we find a good solution
    divisors_desc = sorted(divisors, reverse=True)
    
    # Iterate through each divisor
    for idx, d in enumerate(divisors_desc):
        # Check deadline periodically
        if time.time() > _DEADLINE - 5:
            break
        
        # Iterate sums backwards to ensure each divisor is used at most once
        # We only need to update sums from max_reachable down to 0 (adding d)
        # So new sums range from d to max_reachable + d
        new_max = min(_TARGET, max_reachable + d)
        
        # Check deadline more frequently in the inner loop
        if idx % 10 == 0 and time.time() > _DEADLINE - 5:
            break
            
        for s in range(new_max, d - 1, -1):
            prev = dp[s - d]
            if prev != INF and prev + 1 < dp[s]:
                dp[s] = prev + 1
                parent[s] = d
        
        max_reachable = new_max
        
        # Early termination: if we've reached the target, we can't do better
        # than what we have, but we should continue to see if we can improve
        # Actually, we can't early terminate because a later divisor might
        # allow a smaller cardinality solution
        
    if dp[_TARGET] != INF:
        # Reconstruct the solution
        result = []
        s = _TARGET
        valid = True
        while s > 0:
            d = parent[s]
            if d == -1:
                valid = False
                break
            result.append(d)
            s -= d
        
        if valid and sum(result) == _TARGET and len(set(result)) == len(result):
            return result
    
    # Fallback: Greedy solution
    return _greedy_solve(divisors, _TARGET)
