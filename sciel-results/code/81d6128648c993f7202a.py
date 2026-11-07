import time
import math
from functools import reduce

def construct():
    """
    Finds 10 distinct positive integers <= 1000000, each divisible by its corresponding divisor in [6,10,14,15,21,22,26,33,35,39],
    such that max(reps) - min(reps) is minimized.
    """
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 85  # Safety margin under 90s limit
    
    best_span = float('inf')
    best_result = None
    
    # Order divisors by decreasing value for backtracking (constrained first)
    order = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    
    # Generate candidate L values: multiples of any divisor up to max_val
    candidate_Ls = set()
    for d in divisors:
        candidate_Ls.update(range(d, max_val + 1, d))
    sorted_Ls = sorted(candidate_Ls)
    
    for L in sorted_Ls:
        if time.time() > deadline:
            break
        
        # Try to find an assignment with span < best_span
        S_limit = best_span - 1 if best_span < float('inf') else 1000
        
        if S_limit < 0:
            continue
        
        reps = [0] * n
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            d = divisors[i]
            m = L
            rem = m % d
            if rem != 0:
                m += d - rem
            upper = L + S_limit
            while m <= upper:
                if m not in used:
                    used.add(m)
                    reps[i] = m
                    if backtrack(idx + 1):
                        return True
                    used.remove(m)
                m += d
            return False
        
        if backtrack(0):
            span = max(reps) - min(reps)
            if span < best_span:
                best_span = span
                best_result = reps[:]
                if best_span <= 5:
                    break
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]
