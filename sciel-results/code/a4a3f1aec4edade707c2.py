import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    MAX_DENOM = 10000000
    MIN_DENOM = 5
    
    # Precompute cumulative sums for quick bound checking
    # For a given starting denominator d and depth, we need sum(1/(d+i) for i in range(depth))
    # We'll use a lookup table for small depths and approximation for larger ones
    
    def ceil_div(a, b):
        """Ceiling division for positive integers a, b."""
        return (a + b - 1) // b
    
    # Memoization for DFS states: (rem_num, rem_den, min_d, depth) -> result or None
    memo = {}
    
    def dfs(rem, min_d, depth, path):
        """
        rem: remaining sum as Fraction
        min_d: minimum allowed denominator for the next term
        depth: number of terms left to choose
        path: list of chosen denominators so far
        """
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if rem == 0:
                return path
            return None
        
        if rem == 0:
            return None
        
        if rem < 0:
            return None
        
        # Base case optimization: if only 1 term left, check if rem is a unit fraction
        if depth == 1:
            if rem.numerator == 1:
                d = rem.denominator
                if d >= min_d and d <= MAX_DENOM:
                    return path + [d]
            return None
        
        # Calculate lower bound for the next denominator d
        # We need 1/d <= rem => d >= 1/rem
        # Also d >= min_d
        lower_bound = max(min_d, ceil_div(rem.denominator, rem.numerator))
        
        d = lower_bound
        
        # Pre-calculate float rem for faster bounds
        rem_float = float(rem)
        
        while True:
            if time.time() > deadline:
                return None
            
            if d > MAX_DENOM:
                break
            
            # Optimization: Use float approximation for max sum bound
            # This is much faster than summing Fractions
            # max sum of depth terms starting from d is approx sum(1/(d+i))
            # We can compute this quickly.
            # For small depth, loop is fine.
            max_sum_float = 0.0
            for i in range(depth):
                max_sum_float += 1.0 / (d + i)
                
            # If even the max possible sum (using smallest denominators) is less than rem, 
            # we can't reach rem. 
            # Note: float(rem) might have precision issues, so add small epsilon.
            if max_sum_float < rem_float - 1e-12:
                break

            # Pick d
            new_rem = rem - Fraction(1, d)
            
            if new_rem < 0:
                d += 1
                continue
                
            if new_rem == 0:
                if depth == 1:
                    return path + [d]
                else:
                    d += 1
                    continue

            # Optimization: Check if remaining depth-1 terms can sum to new_rem
            # If not, and since increasing d makes new_rem larger and max sum smaller, we can break.
            if depth > 1:
                max_next_sum = 0.0
                # Approximate max sum of depth-1 terms starting from d+1
                # This loop is small (depth <= 8)
                for i in range(depth - 1):
                    max_next_sum += 1.0 / (d + 1 + i)
                
                if max_next_sum < float(new_rem) - 1e-12:
                    break

            new_min_d = d + 1
            new_depth = depth - 1
            new_path = path + [d]
            
            res = dfs(new_rem, new_min_d, new_depth, new_path)
            if res is not None:
                return res
            
            d += 1
            
        return None

    # Search for minimal k
    # We determined k=7 is impossible (max sum < 1).
    # Start from k=8.
    
    for k in range(8, 257):
        # Quick check if max possible sum < 1 using floats
        max_sum_float = sum(1.0 / (MIN_DENOM + i) for i in range(k))
        if max_sum_float < 1.0:
            continue

        result = dfs(Fraction(1), MIN_DENOM, k, [])
        if result is not None:
            return result
            
    # Fallback: Greedy algorithm
    def greedy(rem, min_d):
        path = []
        while rem > 0:
            if time.time() > deadline:
                return None
            # Find smallest d >= min_d such that 1/d <= rem
            d = max(min_d, ceil_div(rem.denominator, rem.numerator))
            if d > MAX_DENOM:
                return None
            path.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
        return path

    return greedy(Fraction(1), MIN_DENOM)
