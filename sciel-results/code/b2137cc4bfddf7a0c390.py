import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    
    # Fallback solution from seed to ensure we always return something valid
    def get_fallback():
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

    best_solution = get_fallback()
    
    # We want to minimize the largest denominator.
    # Lower bound for largest denominator D:
    # Since 1/d1 <= 5/31, d1 >= 7.
    # If d1=7, rem=4/217.
    # If we use d2, d3..., the max denominator will be at least something.
    # Let's just iterate D from a small value upwards.
    
    # Precompute harmonic sums for pruning?
    # sum(1/i for i in range(a, b+1))
    # For small b, just calculate on the fly or cache.
    
    def max_possible_sum(min_d, max_d):
        # Approximate or exact? Exact is better for pruning.
        # But summing 1/i is fast for small ranges.
        s = Fraction(0, 1)
        for i in range(min_d, max_d + 1):
            s += Fraction(1, i)
        return s

    def search(r, min_d, max_d, path):
        if r == 0:
            return path
        if r < 0 or len(path) >= 256:
            return None
        if min_d > max_d:
            return None
            
        # Pruning: Check if max possible sum from min_d to max_d is less than r
        # Optimization: If max_d - min_d is large, this loop is slow.
        # But we expect max_d to be small for the optimal solution.
        # If max_d is large, we might need a better approximation.
        # Let's add a check: if time is running out, return None.
        if time.time() - start_time > deadline:
            return None
            
        # Estimate max sum. 
        # If range is small, calculate exactly.
        if max_d - min_d < 1000:
            max_sum = max_possible_sum(min_d, max_d)
            if max_sum < r:
                return None
        else:
            # Approximation: integral of 1/x from min_d to max_d
            # ln(max_d) - ln(min_d)
            # If ln(max_d/min_d) < float(r), then max_sum < r is likely.
            # This is a weak pruning for large ranges, but we expect small ranges for optimal D.
            pass

        # Determine start of loop
        # We need 1/d <= r => d >= 1/r
        d_start = max(min_d, (1 // r.numerator) * r.denominator) 
        # ceil(1/r) = ceil(r.denominator / r.numerator)
        d_start = max(min_d, (r.denominator + r.numerator - 1) // r.numerator)
        
        if d_start > max_d:
            return None

        for d in range(d_start, max_d + 1):
            if time.time() - start_time > deadline:
                return None
            new_r = r - Fraction(1, d)
            res = search(new_r, d + 1, max_d, path + [d])
            if res is not None:
                return res
        return None

    # Iterate D from lower bound
    # Lower bound: d1 >= 7.
    # Let's start D from 7.
    # If D=7, only {7} is possible, sum=1/7 != 5/31.
    # We can start D from 10.
    
    D = 10
    while D <= 10000000:
        # Check time
        if time.time() - start_time > deadline:
            break
            
        res = search(target, 1, D, [])
        if res is not None:
            # Found a solution with max denominator D
            # Verify distinctness and sum (should be guaranteed by search)
            # Verify range
            if all(1 <= x <= 10000000 for x in res) and len(set(res)) == len(res):
                if sum(Fraction(1, x) for x in res) == target:
                    return res
            # If verification fails, something is wrong, but search logic should prevent it.
            # Return res anyway if it looks valid.
            return res
            
        D += 1
        
    # If no better solution found, return fallback
    return best_solution
