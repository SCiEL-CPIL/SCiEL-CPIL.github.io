import time
import bisect
from math import factorial

_START = time.time()
_DEADLINE = _START + 55.0  # Safety margin before 60s limit

def construct():
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set()
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.add(val)
    
    valid_divs = sorted(divisors)
    n_divs = len(valid_divs)
    divisor_set = set(valid_divs)  # For O(1) lookup
    
    # Initial known solution to set a tight bound
    best_factors = [22, 21, 25, 18, 36, 64]
    best_max = 64
    
    def time_up():
        return time.time() > _DEADLINE

    def dfs(last_val, remaining, current_max, count, current_factors):
        nonlocal best_max, best_factors
        
        if time_up():
            return
        
        if remaining == 1:
            if count >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        if count >= 12:
            return
            
        # If current max is already >= best_max, we can't improve
        if current_max >= best_max:
            return
        
        # Quick check: can remaining itself be a single valid factor?
        # It must be > last_val, > 12, < best_max, and a divisor of target (it is, since remaining divides target)
        if remaining > last_val and remaining > 12 and remaining < best_max:
            # Check distinctness: remaining not in current_factors
            if remaining not in current_factors:
                new_max = max(current_max, remaining)
                if new_max < best_max:
                    best_max = new_max
                    best_factors = current_factors + [remaining]
                return  # No point continuing to split remaining if we found a valid single-factor completion
        
        # Pruning: the remaining value must be expressible as a product of (12 - count) distinct factors, each > last_val and > 12.
        # The smallest possible product of k distinct integers all > max(last_val, 12) is (max(last_val,12)+1)*(max(last_val,12)+2)*...
        # But more importantly: remaining must be >= the product of the smallest possible factors.
        # Actually, a simpler bound: the next factor d must satisfy d <= remaining / (product of smallest possible remaining factors)
        # But let's use: if we need at least 1 more factor, that factor must be <= remaining.
        # And it must be > last_val. So remaining must be > last_val.
        if remaining <= last_val:
            return
        
        # Lower bound on the maximum factor: 
        # If we have 'count' factors so far and need at least 1 more, the max will be at least max(current_max, next_factor).
        # The next factor must be > last_val. So max >= max(current_max, last_val + 1).
        # But we already check current_max >= best_max.
        
        # The next factor must be > last_val and > 12
        min_d = max(last_val + 1, 13)
        
        # The next factor must be <= remaining (to allow division)
        # And < best_max (to allow improvement)
        max_d = min(remaining, best_max - 1)
        
        if min_d > max_d:
            return
        
        # Additional pruning: remaining / min_d must be >= 1, which is already ensured.
        # Also, the remaining after dividing by d must be factorable into distinct factors > d.
        # A quick necessary condition: remaining / d must be > d (if we need at least one more factor after d)
        # OR remaining / d == 1 (if d is the last factor, already handled above)
        # So if remaining / d <= d and remaining / d != 1, then no valid continuation.
        # This means: for d to work, we need remaining // d > d OR remaining // d == 1.
        # i.e., remaining > d*d OR remaining == d (but remaining > d since min_d <= max_d <= remaining)
        # Actually remaining > d is guaranteed since d <= max_d <= remaining.
        # If remaining == d, then d is the last factor, handled by the quick check above.
        # So for the loop, we need remaining // d > d, i.e., remaining > d*d.
        # This gives us: d < sqrt(remaining) approximately, unless d is the last factor.
        # But d could be the last factor only if remaining == d, which means d = remaining.
        # So in the loop, we can skip d values where remaining > d and remaining // d <= d.
        # That is, skip d where d < remaining <= d*d.
        
        # Find range of divisors in [min_d, max_d]
        start_idx = bisect.bisect_left(valid_divs, min_d)
        end_idx = bisect.bisect_right(valid_divs, max_d)
        
        for i in range(start_idx, end_idx):
            if time_up():
                return
                
            d = valid_divs[i]
            
            # d must divide remaining
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            
            # If new_remaining == 1, d is the last factor - but we already checked if remaining is valid.
            # Since d < remaining (because d <= max_d and if d == remaining, new_remaining == 1)
            # Actually if d == remaining, new_remaining == 1. This case should have been caught by the quick check.
            # But let's handle it: if new_remaining == 1, then d is a valid last factor.
            if new_remaining == 1:
                new_max = max(current_max, d)
                if new_max < best_max:
                    best_max = new_max
                    best_factors = current_factors + [d]
                continue
            
            # If new_remaining <= d, then the next factor after d must be > d, but new_remaining <= d
            # means no valid factor > d can divide new_remaining (since the factor must be <= new_remaining).
            # So this branch is dead.
            if new_remaining <= d:
                continue
            
            new_max = max(current_max, d)
            
            # If new_max >= best_max, we can break because valid_divs is sorted
            if new_max >= best_max:
                break
                
            current_factors.append(d)
            
            dfs(d, new_remaining, new_max, count + 1, current_factors)
            
            current_factors.pop()

    dfs(0, target, 0, 0, [])
    
    return best_factors
