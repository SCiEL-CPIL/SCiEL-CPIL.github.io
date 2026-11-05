import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # First, get the baseline solution from the seed program logic
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            baseline_solution = reached[target]
            break
    else:
        baseline_solution = []
        
    if not baseline_solution:
        raise RuntimeError("no baseline representation")
        
    best_solution = baseline_solution
    min_terms = len(baseline_solution)
    
    # Try to find a better solution with DFS
    def dfs(remaining, min_denom, current_set, depth):
        nonlocal best_solution, min_terms, deadline
        
        if time.time() > deadline:
            return
        
        if depth >= min_terms:
            return
            
        if remaining == 0:
            if len(current_set) < min_terms:
                min_terms = len(current_set)
                best_solution = list(current_set)
            return
            
        if remaining < 0:
            return
            
        if Fraction(1, min_denom) > remaining:
            return
            
        max_d = min(10000000, int(1 / remaining) + 100)
        if max_d < min_denom:
            return
            
        for d in range(min_denom, max_d + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            current_set.append(d)
            dfs(new_remaining, d + 1, current_set, depth + 1)
            current_set.pop()
            
    dfs(Fraction(1, 1), 5, [], 0)
    
    return best_solution

