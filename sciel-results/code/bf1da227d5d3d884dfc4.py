import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, 5s margin before 60s hard kill
    
    N = 1000000
    min_val = 10000
    max_val = N + 1  # need up to 1000001
    
    # Sieve for Largest Prime Factor using array for speed
    lpf = [1] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
    
    if time.time() > deadline:
        return min_val
    
    # Compute score for each n: max(lpf[n], lpf[n+1])
    # We want to find n in [min_val, N] that minimizes this score.
    
    # First pass: find the minimum score achievable
    best_score = float('inf')
    for n in range(min_val, N):
        s = max(lpf[n], lpf[n+1])
        if s < best_score:
            best_score = s
    
    if time.time() > deadline:
        return min_val
    
    # The minimum score is best_score. Now find the smallest n achieving it.
    # But we want to minimize the raw value of n(n+1), which means minimizing n.
    # Wait, the task says "Minimize the largest prime factor of n(n+1)".
    # And "The evaluator measures the raw value (lower is better)".
    # So we want to minimize max(lpf[n], lpf[n+1]).
    # If multiple n achieve the same minimum, we should return the one with smallest n? 
    # No, the score is seed/value where value is the largest prime factor.
    # So we just need to minimize the largest prime factor.
    # If there are ties, any n with that minimum largest prime factor gives the same score.
    # Let's find all n with the minimum score and return the first one (smallest n).
    
    best_n = min_val
    for n in range(min_val, N):
        s = max(lpf[n], lpf[n+1])
        if s == best_score:
            best_n = n
            break  # first one (smallest n) with minimum score
    
    return best_n
