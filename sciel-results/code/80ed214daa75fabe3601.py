import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    N = math.factorial(12)  # 479001600
    
    # Global best
    best_max = float('inf')
    best_factors = []
    
    # Precompute all divisors of N
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = [1]
    primes_exp = [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]
    for p, e in primes_exp:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    
    # Filter: > 12 and sort
    valid_divisors = sorted([d for d in divisors if d > 12])
    valid_set = set(valid_divisors)
    
    # Helper to find divisors of a number 'm' from the precomputed list
    # This is faster than generating divisors from scratch if we filter the global list
    # But filtering the global list of ~1000 items is very fast.
    # We can just iterate valid_divisors and check if m % d == 0
    
    def get_valid_divisors(m, min_val):
        """Return sorted list of divisors of m from valid_divisors that are >= min_val."""
        res = []
        # Optimization: only check divisors up to sqrt(m) if we want to avoid duplicates? 
        # No, we just want all divisors d such that m%d==0 and d>=min_val.
        # Since valid_divisors is sorted, we can break early if d > m (but d is divisor so d<=m).
        # We can also break if d*d > m? No, we need all divisors.
        # However, if d > m, it's not a divisor.
        for d in valid_divisors:
            if d < min_val:
                continue
            if d > m:
                break
            if m % d == 0:
                res.append(d)
        return res

    def dfs(remaining_target, last_factor, depth, target_k, path):
        nonlocal best_max, best_factors
        
        if time.time() - start_time > deadline:
            return False
        
        rem = target_k - depth
        
        if rem == 0:
            if remaining_target == 1:
                max_f = path[-1] if path else 0
                if max_f < best_max:
                    best_max = max_f
                    best_factors = list(path)
            return True
        
        if rem == 1:
            x = remaining_target
            if x > last_factor and x > 12:
                if x < best_max:
                    best_max = x
                    best_factors = path + [x]
            return True
        
        # Lower bound for next factor x:
        # x > last_factor and x > 12
        lower_bound = max(last_factor + 1, 13)
        
        # Upper bound for next factor x:
        # 1. x must be < best_max
        # 2. The product of the 'rem-1' smallest possible distinct integers > x 
        #    must be <= remaining_target / x
        #    i.e., (x+1)*(x+2)*...*(x+rem-1) <= remaining_target / x
        #    x * (x+1) * ... * (x+rem-1) <= remaining_target
        
        x_upper = min(best_max - 1, remaining_target)
        
        # Tighten x_upper using the minimum product constraint
        # We want the largest x such that x * (x+1) * ... * (x+rem-1) <= remaining_target
        # Since x is small (relative to N), we can binary search or just iterate down from a reasonable bound
        # But x is at least lower_bound.
        # Let's estimate max x. (x)^(rem) <= remaining_target => x <= remaining_target^(1/rem)
        max_possible_x = int(remaining_target ** (1.0 / rem)) + 2
        x_upper = min(x_upper, max_possible_x)
        
        if x_upper < lower_bound:
            return True
        
        # Get candidates
        candidates = get_valid_divisors(remaining_target, lower_bound)
        
        for x in candidates:
            if time.time() - start_time > deadline:
                return False
            
            if x > x_upper:
                break
            
            # Check if x can be part of a solution that beats best_max
            # The remaining target is rem_target = remaining_target // x
            # We need to form 'rem-1' distinct factors > x with product rem_target.
            # All these factors must be < best_max.
            # Max product of 'rem-1' distinct integers < best_max:
            # (best_max-1) * (best_max-2) * ... * (best_max - (rem-1))
            # If rem_target > this max product, we can't beat best_max.
            
            max_rem_prod = 1
            for i in range(1, rem):
                max_rem_prod *= (best_max - i)
                if max_rem_prod > remaining_target // x:
                    break
            
            if remaining_target // x > max_rem_prod:
                # Even with the largest possible factors below best_max, we can't reach the target?
                # Wait, if rem_target is LARGE, we might need LARGER factors?
                # No, the constraint is factors < best_max.
                # If the required product rem_target is greater than the max possible product of factors < best_max,
                # then it's impossible to form rem_target with factors < best_max.
                # So we skip.
                continue
            
            # Min product of 'rem-1' distinct integers > x:
            # (x+1) * (x+2) * ... * (x+rem-1)
            min_rem_prod = 1
            for i in range(1, rem):
                min_rem_prod *= (x + i)
                if min_rem_prod > remaining_target // x:
                    break
            
            if remaining_target // x < min_rem_prod:
                # The smallest possible product is already too big.
                # This means x is too small? No, x is the current factor.
                # If min_rem_prod > rem_target, it means we can't fit the remaining factors.
                # Since x is increasing in the loop, smaller x means larger remaining factors?
                # Actually, if x is small, rem_target is large.
                # If x is large, rem_target is small.
                # Here we are iterating x in increasing order.
                # If min_rem_prod > rem_target, it implies the remaining factors must be smaller than x+1, which is impossible.
                # So we can break?
                # Let's check: if x increases, rem_target decreases. min_rem_prod increases (since x increases).
                # So if it fails for x, it might fail for larger x too?
                # min_rem_prod grows as x^rem. rem_target shrinks as 1/x.
                # So yes, if it fails here, it will fail for all larger x.
                break
            
            path.append(x)
            dfs(remaining_target // x, x, depth + 1, target_k, path)
            path.pop()
            
            if best_max <= 15:
                return True
        
        return True

    # Try k values.
    # Geometric mean for k=7 is ~14.5.
    # For k=8, ~12.7 (very tight, factors > 12).
    # For k=6, ~28.
    
    # Start with k=7 as it's a good balance.
    for k in [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 1]:
        if time.time() - start_time > deadline:
            break
        dfs(N, 12, 0, k, [])
        if best_max <= 15:
            break
    
    if not best_factors:
        # Fallback: return a valid solution even if not optimal, to avoid crashing
        # A simple valid solution: [13, 14, ...] might not work easily.
        # Just return N if nothing found (though N > 12, so valid count 1)
        best_factors = [N]
        
    return best_factors
