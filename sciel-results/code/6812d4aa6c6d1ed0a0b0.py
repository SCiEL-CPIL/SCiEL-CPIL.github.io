import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85  # safety margin before 90s kill

def construct():
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of target that are > 12
    divisors = set()
    for a in range(11):  # 2^a, a=0..10
        for b in range(6):  # 3^b, b=0..5
            for c in range(3):  # 5^c, c=0..2
                for d in range(2):  # 7^d, d=0..1
                    for e in range(2):  # 11^e, e=0..1
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.add(val)
    
    valid_divs = sorted(divisors)
    
    # Best solution found so far
    best_max = target
    best_factors = [target]
    
    def dfs(remaining, current_factors, min_factor, max_allowed):
        nonlocal best_max, best_factors
        
        if time.time() > _DEADLINE:
            return
        
        # If we've used up the product
        if remaining == 1:
            if len(current_factors) >= 1:
                current_max = max(current_factors)
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        # Pruning: if we can't beat current best
        if max_allowed <= best_max:
            return
        
        # Try to use fewer factors - check if remaining itself works
        if remaining > 12 and remaining > min_factor:
            # Check if remaining is a divisor (it is by construction)
            candidate_max = max(max(current_factors) if current_factors else 0, remaining)
            if candidate_max < best_max:
                # Check distinctness
                if remaining not in current_factors:
                    dfs(1, current_factors + [remaining], remaining, max_allowed)
        
        # Try splitting remaining into two or more factors
        # We need to find divisors of remaining that are > 12 and > min_factor
        for i, d in enumerate(valid_divs):
            if d <= min_factor:
                continue
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            other = remaining // d
            if other <= 12 and other != 1:
                continue
            if other == d:
                continue  # not distinct
            
            # If other == 1, we're done (handled above)
            if other == 1:
                continue
            
            # Check if other is a valid divisor
            if other in current_factors:
                continue
            
            new_max = max(max(current_factors) if current_factors else 0, d, other)
            if new_max >= best_max:
                continue
            
            # Try this split
            dfs(other, current_factors + [d], max(min_factor, d), max_allowed)
            
            # Also try not using d as a single factor, but this is handled by recursion
            # We need to be careful about order - let's try other as the next factor
            # Actually, let's restructure: pick the smallest factor next
            
        # Better approach: pick the next factor to be at least min_factor
        # and at most some bound
        # Let me restructure the DFS
    
    # Let me rewrite with a cleaner DFS approach
    def dfs2(remaining, current_factors, min_val, max_so_far):
        nonlocal best_max, best_factors
        
        if time.time() > _DEADLINE:
            return
        
        if remaining == 1:
            if current_factors:
                cur_max = max(current_factors)
                if cur_max < best_max:
                    best_max = cur_max
                    best_factors = list(current_factors)
            return
        
        # If remaining is a valid single factor
        if remaining > 12 and remaining >= min_val:
            if remaining not in current_factors:
                new_max = max(max_so_far, remaining)
                if new_max < best_max:
                    dfs2(1, current_factors + [remaining], remaining, new_max)
        
        # Try to split: pick next factor d where min_val <= d <= remaining//2 (to avoid duplicates)
        # and d divides remaining
        # Upper bound for d: we want the resulting max to be < best_max
        # So d < best_max and remaining//d < best_max
        # Thus d > remaining // best_max
        
        lower_bound = max(min_val, remaining // best_max + 1) if best_max > 0 else min_val
        upper_bound = remaining // 2  # to avoid symmetric cases, and d < remaining
        
        if lower_bound > upper_bound:
            return
        
        for d in valid_divs:
            if d < lower_bound:
                continue
            if d > upper_bound:
                break
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            other = remaining // d
            if other <= 12 and other != 1:
                continue
            if other == 1:
                continue
            if other == d:
                continue
            if other in current_factors:
                continue
            if d in current_factors:
                continue
            
            new_max = max(max_so_far, d, other)
            if new_max >= best_max:
                continue
            
            dfs2(other, current_factors + [d], d, new_max)
    
    dfs2(target, [], 13, 0)
    
    return best_factors
