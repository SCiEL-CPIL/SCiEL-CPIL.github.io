from fractions import Fraction

def construct():
    # Find the minimum number of distinct denominators >= 5 with reciprocal sum = 1
    target = Fraction(1, 1)
    
    # The minimum number of terms is at least 8 because sum(1/i for i in 5..11) < 1
    # and sum(1/i for i in 5..12) > 1
    for n in range(8, 257):
        # Quick check: is it possible to reach 1 with n terms starting from 5?
        max_sum = sum(Fraction(1, i) for i in range(5, 5 + n))
        if max_sum < target:
            continue
        
        # Use optimized backtracking search
        result = search(target, 5, n, [])
        if result is not None:
            return result
    
    raise RuntimeError("no representation found")

def search(remaining, min_denom, terms_left, chosen):
    if terms_left == 0:
        return chosen if remaining == 0 else None
    
    if remaining <= 0:
        return None
    
    # Pruning 1: Max possible sum with terms_left terms starting from min_denom
    # If this sum is less than remaining, it's impossible.
    max_possible = sum(Fraction(1, i) for i in range(min_denom, min_denom + terms_left))
    if max_possible < remaining:
        return None
    
    # Determine the range for the next denominator d
    # 1. d >= min_denom
    # 2. 1/d <= remaining  => d >= 1/remaining
    if remaining < 1:
        # ceil(1/remaining)
        min_d_from_sum = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        start_d = max(min_denom, min_d_from_sum)
    else:
        start_d = min_denom
    
    # Iterate d
    # We can break if remaining - 1/d is too large to be formed by the remaining terms
    # Max sum of next (terms_left - 1) terms starting from d+1 is M(d)
    # We need remaining - 1/d <= M(d)
    # If remaining - 1/d > M(d), then for any d' > d, 
    # remaining - 1/d' > remaining - 1/d > M(d) > M(d')
    # So we can break.
    
    for d in range(start_d, 10000000 + 1):
        # Optimization: If d is very large, 1/d is very small.
        # But we have the break condition below.
        
        # Check if 1/d > remaining (should be handled by start_d, but safe)
        if Fraction(1, d) > remaining:
            break # Since d is increasing, 1/d is decreasing? No, 1/d decreases.
            # Wait. If 1/d > remaining, we need smaller 1/d, so larger d.
            # But start_d is calculated to ensure 1/start_d <= remaining.
            # So for d >= start_d, 1/d <= 1/start_d <= remaining.
            # So this condition 1/d > remaining should never be true for d >= start_d.
            # However, if start_d was just min_denom and min_denom was small, 
            # and remaining was small, start_d would be large.
            # So this check is redundant if start_d is correct.
            # But let's keep it safe.
            # Actually, if 1/d > remaining, we should continue to larger d.
            # But my start_d logic ensures 1/start_d <= remaining.
            # So for all d >= start_d, 1/d <= remaining.
            # So this branch is never taken.
            
        new_remaining = remaining - Fraction(1, d)
        if new_remaining < 0:
            break
        
        if terms_left == 1:
            if new_remaining == 0:
                return chosen + [d]
            else:
                # Only one term left, it must be exactly new_remaining
                # So new_remaining must be a unit fraction 1/x with x > d
                # new_remaining = 1/x => x = 1/new_remaining
                # x must be integer and x > d
                if new_remaining.numerator == 1:
                    x = new_remaining.denominator
                    if x > d and x <= 10000000:
                        return chosen + [d, x]
                continue
        
        # Pruning 2: Check if new_remaining can be formed by (terms_left - 1) terms starting from d+1
        # Max sum of (terms_left - 1) terms starting from d+1
        k = terms_left - 1
        if k > 0:
            max_next_sum = sum(Fraction(1, i) for i in range(d + 1, d + 1 + k))
            if new_remaining > max_next_sum:
                # This d is too large (1/d too small), so new_remaining is too big
                # For larger d, new_remaining will be even bigger, and max_next_sum smaller
                # So we can break
                break
        
        result = search(new_remaining, d + 1, terms_left - 1, chosen + [d])
        if result is not None:
            return result
            
    return None
