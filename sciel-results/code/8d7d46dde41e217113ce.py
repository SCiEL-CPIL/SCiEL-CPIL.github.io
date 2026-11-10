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
    
    # DP for minimum cardinality subset sum
    # dp[s] = minimum number of divisors to sum to s
    # parent[s] = the last divisor added to reach s with the current optimal count
    # The 1D DP with reverse iteration ensures 0/1 usage.
    
    INF = 10**9
    dp = [INF] * (_TARGET + 1)
    parent = [-1] * (_TARGET + 1)
    
    dp[0] = 0
    
    # Iterate through each divisor
    for d in divisors:
        # Check deadline periodically
        if time.time() > _DEADLINE - 5:
            break
        # Iterate sums backwards to ensure each divisor is used at most once
        # We only need to update sums >= d
        for s in range(_TARGET, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
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
