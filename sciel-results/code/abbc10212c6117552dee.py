"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s kill

    best_result = None
    best_count = 257

    def check_time():
        return time.time() > deadline

    def dfs(remaining_num, remaining_den, min_den, current, depth, max_terms):
        nonlocal best_result, best_count
        
        if check_time():
            return False
        
        if remaining_num == 0:
            if len(current) < best_count:
                best_count = len(current)
                best_result = list(current)
            return True
        
        if remaining_num <= 0 or depth + 1 > max_terms:
            return False
        
        # Pruning: max possible sum with remaining terms
        if remaining_num * min_den > remaining_den * (max_terms - depth):
            return False
        
        d_low = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)
        
        # Upper bound: 
        # If we pick d, the remaining sum is R' = R - 1/d.
        # We need to form R' with (max_terms - depth - 1) terms, each <= 1/(d+1).
        # So R' <= (max_terms - depth - 1) / (d+1).
        # R - 1/d <= (max_terms - depth - 1) / (d+1)
        # R <= 1/d + (max_terms - depth - 1) / (d+1)
        # This gives an upper bound on d? No, as d increases, RHS decreases.
        # So if d is too large, RHS becomes small, and R might be larger than RHS.
        # So we need R <= 1/d + (max_terms - depth - 1) / (d+1).
        # Let K_rem = max_terms - depth - 1.
        # R <= 1/d + K_rem / (d+1)
        # R <= (d+1 + K_rem*d) / (d(d+1))
        # R * d * (d+1) <= d+1 + K_rem*d
        # This is a quadratic in d.
        # For simplicity, we can just iterate d and break if the condition fails.
        # But we need a starting high bound.
        # A safe high bound is when 1/d is very small, so R ~ K_rem/(d+1) => d ~ K_rem/R.
        # d_high = int((max_terms - depth - 1) * remaining_den / remaining_num) + 100
        
        K_rem = max_terms - depth - 1
        if K_rem < 0:
            return False
            
        # Estimate d_high
        # R <= K_rem / (d+1) => d+1 <= K_rem / R => d <= K_rem / R - 1
        # R = remaining_num / remaining_den
        # d <= K_rem * remaining_den / remaining_num - 1
        d_high = int(K_rem * remaining_den / remaining_num) + 100
        d_high = min(d_high, 10000000)
        
        if d_high < d_low:
            return False
        
        for d in range(d_low, d_high + 1):
            if check_time():
                return False
            
            # Check if 1/d <= remaining
            if d * remaining_num < remaining_den:
                continue
            
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num == 0:
                current.append(d)
                if len(current) < best_count:
                    best_count = len(current)
                    best_result = list(current)
                current.pop()
                return True
            
            current.append(d)
            dfs(new_num, new_den, d + 1, current, depth + 1, max_terms)
            current.pop()
            
            if best_count <= depth + 1:
                return True
        
        return False

    for k in range(5, 20):
        if check_time():
            break
        dfs(1, 1, 5, [], 0, k)
        if best_result is not None and best_count == k:
            return best_result

    if best_result is not None:
        return best_result

    # Fallback: use the original DP approach
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
