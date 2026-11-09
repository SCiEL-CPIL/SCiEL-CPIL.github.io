import math

def construct():
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We can iterate up to sqrt(target) or just generate from prime factorization
    # Given small size, simple iteration is fine
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
            
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    
    # We want to minimize the largest factor.
    # We iterate the maximum allowed factor 'limit' from smallest to largest.
    # For each 'limit', we check if a valid combination exists using only divisors <= 'limit'.
    
    def dfs(start_idx, remaining, current_factors):
        if remaining == 1:
            # Valid solution found
            # Check count constraint: 1 to 12 factors
            if 1 <= len(current_factors) <= 12:
                return current_factors
            else:
                return None
        
        if len(current_factors) >= 12:
            return None
            
        # Pruning: if the product of all remaining available factors (largest ones) 
        # is less than 'remaining', we can't reach it. 
        # However, calculating that every step is expensive. 
        # Simple pruning: if the largest available divisor in the pool is less than remaining, 
        # and we haven't used any factors yet? No, that's not right.
        # Basic pruning: if start_idx >= len(pool), return None.
        
        for i in range(start_idx, len(pool)):
            d = pool[i]
            if d > remaining:
                break
            if remaining % d == 0:
                res = dfs(i + 1, remaining // d, current_factors + [d])
                if res is not None:
                    return res
        return None

    # Iterate limit
    for limit in valid_divs:
        # Pool is valid_divs up to and including 'limit'
        # Since valid_divs is sorted, we can find the index
        # Or simply slice. Slicing creates a new list, which is fine for small N.
        # To be efficient, we can just pass the full list and use an index limit in DFS.
        # Let's modify DFS to take a max_index.
        
        pass # Placeholder, will implement below properly

    # Re-implementing with max_index to avoid slicing overhead
    def dfs_opt(start_idx, remaining, current_factors, max_idx):
        if remaining == 1:
            if 1 <= len(current_factors) <= 12:
                return current_factors
            return None
        
        if len(current_factors) >= 12:
            return None
            
        if start_idx > max_idx:
            return None
            
        # Pruning: Even if we take the largest possible factor at each step, 
        # can we reach 'remaining'?
        # A simple bound: The product of the largest k factors in the pool must be >= remaining.
        # But we don't know k.
        # Let's just rely on the modulo check and sorted order.
        
        for i in range(start_idx, max_idx + 1):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                res = dfs_opt(i + 1, remaining // d, current_factors + [d], max_idx)
                if res is not None:
                    return res
        return None

    # Find the index of the current limit in valid_divs
    # Since we iterate limit in valid_divs, we can just keep track of index
    for idx, limit in enumerate(valid_divs):
        res = dfs_opt(0, target, [], idx)
        if res is not None:
            return res
            
    # If no solution found (should not happen as [target] is a solution if target > 12)
    # 12! is definitely > 12.
    return [target]
