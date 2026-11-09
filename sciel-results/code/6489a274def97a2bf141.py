import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    def dfs(rem_sum, min_d, remaining_terms, current_list):
        # Check time
        if time.time() > deadline:
            return None
            
        if remaining_terms == 0:
            if rem_sum == 0:
                return current_list
            return None
            
        if remaining_terms == 1:
            # rem_sum must be 1/d for some integer d >= min_d
            # 1/d = rem_sum => d = 1/rem_sum
            # Check if 1/rem_sum is an integer
            # rem_sum is a fraction or float? 
            # Using floats for rem_sum is risky for exactness.
            # But for small denominators, precision is usually okay.
            # Better to use fractions.Fraction or integer arithmetic.
            # Given the constraints, let's use float but check carefully.
            # Or convert rem_sum to fraction.
            pass

    # To be safe with exactness, let's use fractions.Fraction
    from fractions import Fraction
    
    def dfs_exact(rem_sum, min_d, remaining_terms, current_list):
        if time.time() > deadline:
            return None
            
        if remaining_terms == 0:
            if rem_sum == 0:
                return current_list
            return None
            
        if remaining_terms == 1:
            # rem_sum = 1/d => d = 1/rem_sum
            if rem_sum > 0:
                d = 1 / rem_sum
                # Check if d is integer and >= min_d
                if d >= min_d and d == int(d) and int(d) >= 5:
                    return current_list + [int(d)]
            return None

        # Pruning: Max possible sum with remaining_terms terms starting from min_d
        # The smallest denominators are min_d, min_d+1, ...
        # If the sum of reciprocals of the smallest 'remaining_terms' denominators
        # is less than rem_sum, it's impossible.
        # Wait, we are picking the NEXT denominator d.
        # The current rem_sum is what we need to fill with 'remaining_terms' denominators.
        # The smallest possible denominators we can pick are min_d, min_d+1, ..., min_d + remaining_terms - 1.
        # If sum(1/(min_d + i) for i in range(remaining_terms)) < rem_sum, then impossible.
        
        max_possible_sum = 0
        for i in range(remaining_terms):
            max_possible_sum += Fraction(1, min_d + i)
            
        if max_possible_sum < rem_sum:
            return None

        # Iterate d from min_d upwards
        # Upper bound for d: 1/d <= rem_sum => d >= 1/rem_sum
        # Also d cannot be too large if we need to fit remaining terms.
        
        # Lower bound for d:
        # We need 1/d <= rem_sum => d >= ceil(1/rem_sum)
        # Also d >= min_d
        
        # Upper bound for d:
        # Even if we pick the largest possible d, the remaining sum must be fillable.
        # Actually, if d is too large, 1/d is small, rem_sum - 1/d is large.
        # The max sum of the next (remaining_terms - 1) terms is sum(1/(d+1 ... d+remaining_terms-1)).
        # If rem_sum - 1/d > max_next_sum, then d is too small? No.
        # If d is small, 1/d is large, rem_sum - 1/d is small.
        # If d is large, 1/d is small, rem_sum - 1/d is large.
        # The condition for impossibility is:
        # rem_sum - 1/d > sum(1/(d+i) for i in 1..remaining_terms-1)
        # If this holds, d is too LARGE?
        # Let's check.
        # If d is large, 1/d is small. LHS is large.
        # RHS involves 1/(d+1)... which is small.
        # So if LHS > RHS, it means we need a large sum, but the available denominators are too large (small reciprocals).
        # So if this condition holds, d is too large.
        # Since as d increases, LHS increases and RHS decreases, the condition will hold for all larger d.
        # So we can break.
        
        d = max(min_d, (1 // rem_sum) + 1 if rem_sum < 1 else 1)
        # Wait, 1/rem_sum might not be integer.
        # d must satisfy 1/d <= rem_sum => d >= 1/rem_sum.
        # So d_min = ceil(1/rem_sum).
        
        d_min = max(min_d, int(1 / rem_sum) + (1 if 1 % rem_sum != 0 else 0))
        # Careful with Fraction division.
        # 1 / rem_sum is a Fraction.
        # ceil of a Fraction p/q is (p + q - 1) // q.
        
        inv_rem = 1 / rem_sum
        d_min = max(min_d, int(inv_rem) + (1 if inv_rem != int(inv_rem) else 0))
        
        for d in range(d_min, 10000001):
            if time.time() > deadline:
                return None
                
            if Fraction(1, d) > rem_sum:
                break
                
            # Pruning check
            if remaining_terms > 1:
                rem_new = rem_sum - Fraction(1, d)
                # Max sum of next remaining_terms - 1 terms starting from d+1
                max_next_sum = 0
                for i in range(1, remaining_terms):
                    max_next_sum += Fraction(1, d + i)
                    
                if rem_new > max_next_sum:
                    # d is too large (or rather, we can't fill the remaining sum with larger denominators)
                    # Actually, if rem_new > max_next_sum, it means we need MORE sum than the smallest denominators can provide.
                    # This happens if d is too SMALL?
                    # Let's re-verify.
                    # If d is small, 1/d is large. rem_new is small.
                    # max_next_sum is based on d+1, which is also small, so max_next_sum is large.
                    # So rem_new < max_next_sum is likely.
                    # If d is large, 1/d is small. rem_new is large.
                    # max_next_sum is based on d+1, which is large, so max_next_sum is small.
                    # So rem_new > max_next_sum is likely.
                    # So if rem_new > max_next_sum, d is too LARGE.
                    # Wait, if d is too large, we should break?
                    # Yes, because increasing d further makes rem_new larger and max_next_sum smaller.
                    break
                    
            res = dfs_exact(rem_new, d + 1, remaining_terms - 1, current_list + [d])
            if res is not None:
                return res
                
        return None

    # Try 8 terms first
    for k in range(8, 15):
        if time.time() > deadline:
            break
        res = dfs_exact(Fraction(1, 1), 5, k, [])
        if res is not None:
            return res
            
    # Fallback to seed logic if no small solution found (should not happen)
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
            
    raise RuntimeError("no solution found")
