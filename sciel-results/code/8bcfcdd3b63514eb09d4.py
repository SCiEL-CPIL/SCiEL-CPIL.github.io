"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for overhead/evaluation
    
    # We know k must be at least 8 because sum_{i=0}^{6} 1/(5+i) < 1
    # and sum_{i=0}^{7} 1/(5+i) > 1.
    # Let's search for k=8 first, then 9, etc.
    
    for k in range(8, 257):
        if time.time() > deadline:
            break
            
        solution = search_egyptian(k)
        if solution:
            return solution
            
    # Fallback: return a known valid solution if search fails or times out
    # The seed program's DP approach on 2520 finds a solution.
    # We can just return a hardcoded valid solution to be safe, 
    # or run the original DP if we have time.
    # But the original DP is slow to write from scratch here.
    # Let's just return a hardcoded solution with k=8 if we found one.
    # If we didn't, we should probably run a simple greedy or DP.
    
    # For robustness, if the search loop finishes without finding anything (unlikely for k=8),
    # we fall back to a simple construction.
    # However, k=8 solution is almost certainly found instantly.
    
    # Fallback to original logic if k=8..256 search fails (should not happen)
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")

def search_egyptian(k):
    # Backtracking search for k distinct denominators >= 5 summing to 1
    # Use fractions for exact arithmetic
    from fractions import Fraction
    
    def backtrack(remaining_terms, min_denom, current_sum, current_denoms):
        if remaining_terms == 0:
            if current_sum == 1:
                return current_denoms
            return None
        
        # Pruning:
        # 1. If current sum > 1, fail
        if current_sum > 1:
            return None
            
        # 2. Max possible sum with remaining terms
        # Max sum is sum_{i=0}^{remaining_terms-1} 1/(min_denom + i)
        max_possible = sum(Fraction(1, min_denom + i) for i in range(remaining_terms))
        if current_sum + max_possible < 1:
            return None
            
        # 3. Min possible sum with remaining terms
        # Min sum is sum_{i=0}^{remaining_terms-1} 1/(10**7 - i) approx 0
        # This is not very useful for pruning > 1, but current_sum > 1 covers it.
        
        # Determine range for next denominator
        # 1/d <= 1 - current_sum
        # d >= 1 / (1 - current_sum)
        # Also d >= min_denom
        
        if current_sum >= 1:
            return None
            
        target_remaining = 1 - current_sum
        # d must satisfy 1/d <= target_remaining
        # d >= 1/target_remaining
        min_d = max(min_denom, int(1 / target_remaining) + 1)
        
        # Upper bound for d:
        # 1/d > target_remaining / remaining_terms
        # d < remaining_terms / target_remaining
        max_d = int(remaining_terms / target_remaining)
        
        for d in range(min_d, max_d + 1):
            # Check if d is large enough to keep sum <= 1
            # Actually the loop bound handles this roughly.
            # We must ensure d is distinct (guaranteed by min_denom = prev + 1)
            
            new_sum = current_sum + Fraction(1, d)
            if new_sum > 1:
                continue
                
            # Recurse
            result = backtrack(remaining_terms - 1, d + 1, new_sum, current_denoms + [d])
            if result:
                return result
                
        return None

    return backtrack(k, 5, Fraction(0), [])

# EVOLVE-BLOCK-END
