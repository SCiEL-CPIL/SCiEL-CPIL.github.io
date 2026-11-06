"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    target_num, target_den = 5, 31
    
    # Get initial upper bound via greedy
    def greedy_max_den(num, den):
        r_num, r_den = num, den
        max_d = 0
        while r_num > 0:
            d = (r_den + r_num - 1) // r_num
            if d > max_d:
                max_d = d
            new_num = r_num * d - r_den
            new_den = r_den * d
            if new_num <= 0:
                break
            g = gcd(new_num, new_den)
            r_num, r_den = new_num // g, new_den // g
        return max_d
    
    upper = greedy_max_den(target_num, target_den)
    
    # Feasibility check with memoization and pruning
    def can_represent(t_num, t_den, max_denom):
        failed = set()
        
        def dfs(r_num, r_den, min_d, depth):
            if r_num == 0:
                return True
            if depth >= 256 or time.time() > deadline:
                return False
            
            state = (r_num, r_den, min_d)
            if state in failed:
                return False
            
            lb = (r_den + r_num - 1) // r_num
            if lb < min_d:
                lb = min_d
            
            if lb > max_denom:
                failed.add(state)
                return False
            
            # Pruning: even with optimal choices, can we reach the sum?
            # The maximum sum we can achieve with remaining terms starting from min_d:
            # 1/min_d + 1/(min_d+1) + ... but this is complex. 
            # Simpler: if 1/lb + 1/(lb+1) + ... < r_num/r_den, we need more but they're all >= lb
            # Actually, the minimum possible next term is 1/max_denom, and we have at most (256-depth) terms
            # But this is a weak bound. Let's just proceed.
            
            for d in range(lb, max_denom + 1):
                # 1/d must be <= r_num/r_den
                if d * r_num < r_den:
                    continue  # 1/d < remainder, but we need sum to equal, so this is ok actually
                # Wait: we need 1/d <= r_num/r_den, i.e., d >= r_den/r_num
                # lb already ensures this, so all d in [lb, max_denom] are valid
                new_num = r_num * d - r_den
                if new_num < 0:
                    continue
                new_den = r_den * d
                g = gcd(new_num, new_den)
                nr_num, nr_den = new_num // g, new_den // g
                if dfs(nr_num, nr_den, d + 1, depth + 1):
                    return True
            
            failed.add(state)
            return False
        
        return dfs(t_num, t_den, 1, 0)
    
    # Binary search for minimum max denominator
    lo, hi = 1, upper
    best = upper
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        if can_represent(target_num, target_den, mid):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Find actual representation
    def find_repr(t_num, t_den, max_denom):
        memo = {}
        
        def dfs(r_num, r_den, min_d, depth, path):
            if r_num == 0:
                return list(path)
            if depth >= 256 or time.time() > deadline:
                return None
            
            state = (r_num, r_den, min_d)
            if state in memo:
                return memo[state]
            
            lb = (r_den + r_num - 1) // r_num
            if lb < min_d:
                lb = min_d
            
            if lb > max_denom:
                memo[state] = None
                return None
            
            for d in range(lb, max_denom + 1):
                new_num = r_num * d - r_den
                if new_num < 0:
                    continue
                new_den = r_den * d
                g = gcd(new_num, new_den)
                nr_num, nr_den = new_num // g, new_den // g
                path.append(d)
                result = dfs(nr_num, nr_den, d + 1, depth + 1, path)
                path.pop()
                if result is not None:
                    memo[state] = result
                    return result
            
            memo[state] = None
            return None
        
        return dfs(t_num, t_den, 1, 0, [])
    
    result = find_repr(target_num, target_den, best)
    
    if result is None or len(result) == 0:
        # Fallback
        r_num, r_den = target_num, target_den
        values = []
        while r_num > 0:
            d = (r_den + r_num - 1) // r_num
            values.append(d)
            new_num = r_num * d - r_den
            new_den = r_den * d
            if new_num <= 0:
                break
            g = gcd(new_num, new_den)
            r_num, r_den = new_num // g, new_den // g
        return values
    
    return result

# EVOLVE-BLOCK-END
