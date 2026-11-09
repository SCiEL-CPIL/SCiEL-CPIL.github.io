import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # 1. Calculate 12!
    target = math.factorial(12)  # 479001600
    
    # 2. Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set()
    for a in range(11):      # 2^a
        for b in range(6):   # 3^b
            for c in range(3): # 5^c
                for d in range(2): # 7^d
                    for e in range(2): # 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)
    
    # Filter: must be > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Initial best solution: single factor [target]
    best_max = target
    best_factors = [target]
    
    # Try to find a better initial solution using a greedy approach
    # Try k factors from 12 down to 2
    def try_greedy(k, max_val):
        """Try to find k distinct divisors > 12, all <= max_val, product = target"""
        if k == 1:
            if target <= max_val and target > 12:
                return [target]
            return None
        
        # The remaining product for k-1 factors must be >= (k-1) smallest possible values
        # and <= (k-1) largest possible values <= max_val
        # Simple bound: each of the k-1 remaining factors must be >= 13 (smallest > 12)
        # and <= max_val
        min_possible = 13 ** (k-1)
        max_possible = max_val ** (k-1)
        
        if target < min_possible or target > max_possible:
            return None
        
        # Try each divisor d <= max_val that divides target
        for d in valid_divs:
            if d > max_val:
                break
            if target % d != 0:
                continue
            remaining = target // d
            if remaining < 13 ** (k-1) or remaining > max_val ** (k-1):
                continue
            sub = try_greedy(k-1, max_val)
            if sub is not None:
                if d not in sub:  # distinctness
                    return [d] + sub
        return None
    
    # Try to find good solutions with various k
    # Start with k=12 and work down
    for k in range(12, 1, -1):
        if time.time() > deadline:
            break
        # Estimate what max_val we might need
        # Each factor roughly target^(1/k)
        est = int(target ** (1.0/k)) + 1
        # Try a range of max_val around the estimate
        for max_val in range(est, est + 200, 5):
            if time.time() > deadline:
                break
            sol = try_greedy(k, max_val)
            if sol is not None and max(sol) < best_max:
                best_max = max(sol)
                best_factors = sol[:]
    
    # Now do a more thorough DFS search to potentially find even better solutions
    # Use the current best_max as the upper bound
    def dfs(remaining, start_idx, current_factors, current_max, k_target):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if 1 <= len(current_factors) <= 12:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = current_factors[:]
            return
        
        # We can have at most 12 factors
        if len(current_factors) >= 12:
            return
        
        # Pruning: if current_max is already >= best_max, no point continuing
        if current_max >= best_max:
            return
            
        # Pruning: if the smallest available factor is already >= best_max
        if start_idx < len(valid_divs):
            if valid_divs[start_idx] >= best_max:
                return
        else:
            return
        
        # Additional pruning: remaining must be achievable by some number of distinct factors > 12
        # Each remaining factor must be > 12 and <= best_max - 1 (to improve)
        # So remaining must be a product of m distinct integers in [13, best_max-1]
        # Lower bound: 13^1 * 14^1 * ... but that's complex. Simple: remaining >= 13
        if remaining < 13:
            return
        
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            
            if new_max >= best_max:
                break
            
            current_factors.append(d)
            dfs(remaining // d, i + 1, current_factors, new_max, k_target)
            current_factors.pop()
            
    dfs(target, 0, [], 0, 0)
    
    return best_factors
