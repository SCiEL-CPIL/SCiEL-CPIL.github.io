import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    MAX_DENOM = 10000000
    MIN_DENOM = 5

    def ceil_div(a, b):
        """Ceiling division for positive integers a, b."""
        return (a + b - 1) // b

    def dfs(rem_frac, rem_float, min_d, depth, path):
        """
        rem_frac: remaining sum as Fraction
        rem_float: remaining sum as float (for fast pruning)
        min_d: minimum allowed denominator for the next term
        depth: number of terms left to choose
        path: list of chosen denominators so far
        """
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if rem_frac == 0:
                return path
            return None
        
        if rem_frac == 0:
            return None

        if rem_float < 0:
            return None

        # Calculate lower bound for the next denominator d
        # We need 1/d <= rem => d >= 1/rem
        # Using float for speed
        lower_bound_float = 1.0 / rem_float if rem_float > 0 else float('inf')
        lower_bound = max(min_d, int(lower_bound_float) + 1)
        # Adjust for floating point precision: ensure 1/lower_bound <= rem
        # If 1/lower_bound > rem, increment
        while Fraction(1, lower_bound) > rem_frac:
            lower_bound += 1
            
        d = lower_bound
        
        while True:
            if time.time() > deadline:
                return None
            
            if d > MAX_DENOM:
                break
            
            # Check if it's even possible to reach 'rem' with 'depth' terms starting from d
            # Max sum is sum(1/(d+i) for i in range(depth))
            # If this is less than rem, break.
            
            # Approximation for speed using float
            # sum(1/(d+i)) is approx integral or just sum for small depth
            # For depth <= 20, exact float sum is fast and accurate enough for pruning
            if depth <= 20:
                max_sum_float = sum(1.0 / (d + i) for i in range(depth))
                if max_sum_float < rem_float:
                    break
            else:
                # Loose bound: depth/d
                if depth / d < rem_float:
                    break

            # Pick d
            # Verify with Fraction to be safe before recursing
            new_rem_frac = rem_frac - Fraction(1, d)
            
            if new_rem_frac < 0:
                d += 1
                continue
                
            if new_rem_frac == 0:
                if depth == 1:
                    return path + [d]
                else:
                    d += 1
                    continue

            new_rem_float = rem_float - 1.0 / d
            
            new_min_d = d + 1
            new_depth = depth - 1
            new_path = path + [d]
            
            res = dfs(new_rem_frac, new_rem_float, new_min_d, new_depth, new_path)
            if res is not None:
                return res
            
            d += 1
            
        return None

    def greedy(rem_frac, min_d):
        """
        Fallback greedy algorithm.
        Guarantees a solution if one exists (which it does for 1 with denom >= 5).
        """
        path = []
        while rem_frac > 0:
            if time.time() > deadline:
                return None
            # Find smallest d >= min_d such that 1/d <= rem
            # 1/d <= rem => d >= 1/rem
            inv_rem = 1 / rem_frac
            d = max(min_d, ceil_div(inv_rem.denominator, inv_rem.numerator))
            
            # Ensure 1/d <= rem
            while Fraction(1, d) > rem_frac:
                d += 1
                
            if d > MAX_DENOM:
                return None
            
            path.append(d)
            rem_frac -= Fraction(1, d)
            min_d = d + 1
            
            # Safety check for infinite loop or error
            if len(path) > 256:
                return None
        return path

    # Search for minimal k
    # We established k >= 6
    # Let's try k from 6 to 256
    # But we should stop if we run out of time.
    
    # Optimization: Check if max sum for k terms < 1
    # If sum(1/(5+i) for i in range(k)) < 1, skip k.
    
    for k in range(6, 257):
        # Quick check if max possible sum < 1
        # Use float for this check
        max_sum_float = sum(1.0 / (5 + i) for i in range(k))
        if max_sum_float < 1.0:
            continue
            
        # If we are getting close to deadline, stop trying new k's?
        # No, if we haven't found a solution for smaller k, we must try this one.
        # But if time is running out, we should break and use greedy.
        
        if time.time() > deadline - 5:
            break
            
        result = dfs(Fraction(1), 1.0, MIN_DENOM, k, [])
        if result is not None:
            return result
            
    # Fallback: Greedy algorithm
    # This guarantees a solution, though not minimal.
    result = greedy(Fraction(1), MIN_DENOM)
    if result is not None:
        return result
        
    # Absolute last resort: return a known valid solution if any?
    # Or just raise? No, must return list.
    # If greedy failed (unlikely), return empty? No, must be valid.
    # Greedy should not fail for 1 with min denom 5.
    # 1 = 1/5 + 1/6 + 1/7 + ... 
    # Greedy will find it.
    
    return []

