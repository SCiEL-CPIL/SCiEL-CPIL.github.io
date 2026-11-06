import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    # We want to minimize the largest denominator.
    # We can iterate K from 1 upwards.
    # For a given K, we check if there is a subset of {1..K} summing to target.
    # To make it faster, we can use a recursive search with pruning.
    
    def dfs(current_sum, last_denom, remaining_terms, path):
        # current_sum: Fraction
        # last_denom: int, last denominator used
        # remaining_terms: int, max terms left to add (we can use 0 to 256)
        # path: list of denominators used
        
        if current_sum == target:
            return path
        
        if current_sum > target:
            return None
            
        # Pruning: if even adding 1/(last_denom+1) + ... doesn't reach target?
        # Actually, we want to reach exactly target.
        # Max possible sum if we add smallest available denominators.
        # Min possible sum if we add largest available denominators? No, we just need to check reachability.
        
        # If we are close to deadline, return None
        if time.time() > deadline:
            return None
            
        # Heuristic: if current_sum is very close to target, 
        # we need 1/d = target - current_sum.
        diff = target - current_sum
        if diff == 0:
            return path
            
        # If diff is negative, handled above.
        
        # Try to add one term that exactly matches diff?
        d_exact = diff.denominator // diff.numerator if diff.numerator != 0 else 0
        # Actually 1/d = diff => d = 1/diff = diff.denominator / diff.numerator
        if diff.numerator != 0:
            d_candidate = diff.denominator // diff.numerator
            if diff.numerator != 0 and diff.denominator % diff.numerator == 0:
                d_candidate = diff.denominator // diff.numerator
                if d_candidate > last_denom and d_candidate <= K:
                    # Check if d_candidate is distinct (it is, since > last_denom)
                    new_path = path + [d_candidate]
                    return new_path
            
        # Search for next denominator
        # d must be > last_denom
        # 1/d <= diff => d >= 1/diff
        min_d = max(last_denom + 1, int(1 / diff) + 1)
        
        # Max d?
        # We need to be able to form the remaining sum with terms > d.
        # If we pick d, new_diff = diff - 1/d.
        # We need new_diff >= 0.
        
        # Also, we need enough "capacity" from smaller denominators? No, we are picking increasing denominators.
        # The remaining sum must be formable by denominators > d.
        
        # Upper bound for d:
        # If we pick d, the maximum sum we can get from subsequent terms (assuming we can pick as many as we want, but distinct)
        # is bounded by sum_{j=d+1}^K 1/j.
        # But we don't know how many terms.
        
        # Let's just iterate d from min_d to K.
        for d in range(min_d, K + 1):
            new_sum = current_sum + Fraction(1, d)
            if new_sum > target:
                break # Since 1/d decreases, further d will be smaller, so sum will be even smaller? 
                # Wait, 1/d decreases as d increases.
                # So if new_sum > target, increasing d will decrease 1/d, so new_sum will decrease.
                # So we should NOT break. We should continue.
                # Actually, if 1/d is too big, we need a bigger d to make 1/d smaller.
                # So we continue.
            
            # Pruning: if new_sum + max_possible_remaining < target, continue?
            # Max possible remaining sum with denominators > d up to K.
            # This is sum_{j=d+1}^K 1/j.
            # We can precompute suffix sums of reciprocals for 1..K.
            
            # Let's just recurse.
            res = dfs(new_sum, d, remaining_terms - 1, path + [d])
            if res:
                return res
                
        return None

    # Precompute suffix sums of reciprocals for pruning?
    # suffix_sum[i] = sum_{j=i}^K 1/j
    
    # We will iterate K from 1 to 200.
    # For each K, we check if a solution exists.
    
    # To speed up, we can check small K first.
    # The seed has max denom 1860.
    # We hope to find one < 1860.
    
    # Let's try K up to 100 first.
    # If not found, expand.
    
    best_solution = None
    
    # We can also try specific small sets.
    # But DFS is general.
    
    # Optimization: 
    # If we fix K, we can use meet-in-the-middle if K is small (< 40).
    # For K > 40, DFS might be slow.
    # But the number of terms is likely small.
    
    # Let's try to find a solution with max denom <= 50.
    # If not, <= 100.
    
    for K in range(1, 200):
        if time.time() > deadline:
            break
            
        # Precompute suffix sums for pruning in DFS
        # suffix[i] = sum_{j=i}^K 1/j
        suffix = [0] * (K + 2)
        for i in range(K, 0, -1):
            suffix[i] = suffix[i+1] + Fraction(1, i)
            
        # Modify DFS to use suffix for pruning
        def dfs_opt(current_sum, last_denom, path):
            if current_sum == target:
                return path
            if current_sum > target:
                return None
            if time.time() > deadline:
                return None
                
            diff = target - current_sum
            if diff < 0:
                return None
                
            # Check if we can complete with one term
            if diff.numerator != 0:
                d_candidate = diff.denominator // diff.numerator
                if diff.denominator % diff.numerator == 0:
                    d_candidate = diff.denominator // diff.numerator
                    if d_candidate > last_denom and d_candidate <= K:
                        return path + [d_candidate]
                        
            # Pruning: if diff > suffix[last_denom+1], we can't reach target
            # because max sum with denominators > last_denom is suffix[last_denom+1]
            if last_denom + 1 <= K:
                if diff > suffix[last_denom+1]:
                    return None
            else:
                # No denominators left
                if diff != 0:
                    return None

            min_d = max(last_denom + 1, int(1 / diff) + 1)
            
            for d in range(min_d, K + 1):
                new_sum = current_sum + Fraction(1, d)
                if new_sum > target:
                    # Since 1/d decreases, new_sum will decrease for larger d.
                    # So we can continue.
                    pass
                
                res = dfs_opt(new_sum, d, path + [d])
                if res:
                    return res
                    
            return None

        res = dfs_opt(Fraction(0), 0, [])
        if res:
            return res
            
    # If no solution found with K < 200, fall back to seed or larger K
    # But we should find one.
    
    # Fallback to seed
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
