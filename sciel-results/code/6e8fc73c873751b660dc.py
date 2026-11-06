import time
from math import gcd

DENOM_BOUND = 10**12

def _find_solution_for_n(n, deadline):
    """Find distinct x,y,z with 1/x+1/y+1/z = 4/n, x<y<z, z<=DENOM_BOUND"""
    nx = n
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    # Try x values in an order likely to find solutions quickly
    # Start with x near n/2 (middle of range), then expand outward
    mid = (x_min + x_max) // 2
    x_candidates = []
    for d in range(0, x_max - x_min + 1):
        if mid - d >= x_min:
            x_candidates.append(mid - d)
        if mid + d <= x_max and d > 0:
            x_candidates.append(mid + d)
    
    for x in x_candidates:
        if time.time() > deadline:
            return None
            
        A = 4 * x - n
        if A <= 0:
            continue
            
        N = nx * x
        
        y_lower = N // A + 1
        if y_lower < x:
            y_lower = x + 1  # ensure x < y for distinctness
        
        y_upper = (2 * N) // A
        
        if y_lower > y_upper:
            continue
        
        for y in range(y_lower, y_upper + 1):
            D = A * y - N
            if D <= 0:
                continue
                
            if (N * y) % D != 0:
                continue
                
            z = (N * y) // D
            
            if z <= y:
                continue
                
            if z > DENOM_BOUND:
                continue
            
            return [x, y, z]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    answers = [None] * 254
    
    # First pass: quick search for all n with tight per-n budget
    # This ensures we get solutions for easy cases
    per_n_budget = 0.3  # 300ms per n for first pass
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline - 10.0:  # Leave 10s for second pass
            break
            
        n_deadline = min(time.time() + per_n_budget, deadline - 5.0)
        found = _find_solution_for_n(n, n_deadline)
        answers[i] = found
    
    # Second pass: retry unsolved n with more time
    remaining = deadline - time.time()
    if remaining > 0.5:
        unsolved_indices = [i for i in range(254) if answers[i] is None]
        per_n_budget2 = max(0.1, remaining / max(1, len(unsolved_indices)))
        
        for i in unsolved_indices:
            if time.time() > deadline - 1.0:
                break
            
            n = i + 3
            n_deadline = min(time.time() + per_n_budget2, deadline - 0.5)
            found = _find_solution_for_n(n, n_deadline)
            if found:
                answers[i] = found
    
    return answers
