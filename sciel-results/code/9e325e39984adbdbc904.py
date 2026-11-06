"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial
    import itertools
    
    N = 12
    target = factorial(N)
    
    # We need distinct integers > N with product exactly target.
    # Minimize the largest factor.
    # The number of factors is variable (1..12).
    
    # Strategy:
    # 1. Find all divisors of target that are > N.
    # 2. We want a subset of distinct divisors D = {d1, d2, ..., dk} such that prod(D) == target.
    # 3. Minimize max(D).
    
    # Note: The factors themselves don't have to be prime. They just have to be integers > 12.
    # For example, 13 * 14 * ... is not required. 13 * 15 * ... is fine as long as product is 12!.
    
    # Let's enumerate possible sets of divisors.
    # Since we minimize the largest factor, we can try increasing upper bounds M.
    # For a given M, check if there exists a subset of distinct divisors in (N, M] whose product is target.
    
    # Get all divisors of target
    divisors = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            if i > N:
                divisors.append(i)
            if i != target // i and target // i > N:
                divisors.append(target // i)
        i += 1
    
    # Sort divisors
    divisors.sort()
    
    # We can use a recursive search (backtracking) to find a subset with product == target.
    # To minimize the largest, we iterate M from smallest possible divisor upwards.
    # Actually, we can just do a DFS that tries to build the product, prioritizing smaller numbers?
    # No, we want to minimize the MAX. So we should try to use small numbers, but we must ensure
    # we don't get stuck.
    
    # Better approach:
    # Iterate through possible maximum values M from the smallest divisor upwards.
    # For each M, consider only divisors <= M.
    # Check if there is a subset of distinct divisors from this set that multiplies to target.
    # The first M for which such a subset exists gives the optimal answer (since we check M in increasing order).
    
    # However, the number of divisors can be large.
    # 12! = 479001600.
    # Number of divisors of 12!:
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Divisors count = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
    # Divisors > 12: a bit less.
    
    # So we have ~790 divisors.
    # We need to find a subset.
    # Since the product is fixed, and numbers are > 12, the number of factors is limited.
    # 13^12 is huge.
    # 13^1 = 13
    # 13^2 = 169
    # 13^3 = 2197
    # 13^4 = 28561
    # 13^5 = 371293
    # 13^6 = 4826809
    # 13^7 = 62748517
    # 13^8 = 815730721 > 12!
    # So max 7 factors if all are 13.
    # If we use larger numbers, even fewer.
    # So k is at most 7 or 8.
    
    # We can use a recursive backtracking search.
    # Sort divisors ascending.
    # Try to pick factors.
    # To minimize the max, we should iterate the maximum allowed value M.
    # For a fixed M, we have a set of allowed divisors S_M = {d in divisors | d <= M}.
    # We need to find a subset of S_M with product target.
    
    # Since we want the global minimum max, we can do this:
    # 1. Generate all divisors > 12.
    # 2. Sort them.
    # 3. For each possible max value M (from the sorted list), run a search to see if a valid subset exists using only values <= M.
    # 4. The first M that works is the answer.
    
    # To speed up the search for a specific M:
    # We can use a recursive function that takes:
    # - current_product
    # - remaining_target (target / current_product)
    # - index in the sorted list of allowed divisors
    # - current_subset
    
    # Optimization:
    # If current_product * (smallest possible next factor) > remaining_target, prune.
    # If current_product > remaining_target, prune.
    # If remaining_target == 1, we found a solution (if we have at least 1 factor? No, product of empty set is 1, but we need factors > 12. So we need at least one factor. If remaining_target becomes 1, it means we've reached the target exactly).
    
    # Actually, let's just use a simple DFS with pruning.
    # Since the number of factors is small (<=8), the depth is small.
    # The branching factor is the number of divisors <= M.
    
    # Let's implement the search for a given M.
    
    def find_subset(target, divisors, max_val):
        # divisors is sorted list of all divisors > 12
        # We only consider divisors <= max_val
        allowed = [d for d in divisors if d <= max_val]
        
        # If product of all allowed divisors < target, impossible? No, we don't have to use all.
        # But if the largest divisor in allowed is < target, and we can't combine enough to reach target?
        
        # Recursive helper
        def dfs(idx, current_prod, current_subset):
            if current_prod > target:
                return None
            if current_prod == target:
                return current_subset
            
            # If we run out of candidates
            if idx >= len(allowed):
                return None
            
            # Pruning: even if we multiply by the largest possible remaining factors, can we reach target?
            # Actually, simpler pruning:
            # If current_prod * allowed[idx] > target, and since allowed is sorted, all subsequent are larger, so we can break?
            # No, we can skip allowed[idx] and try allowed[idx+1].
            
            # We iterate from idx to end
            for i in range(idx, len(allowed)):
                d = allowed[i]
                if current_prod * d > target:
                    break # Since sorted, further d will be larger
                # Try including d
                res = dfs(i + 1, current_prod * d, current_subset + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(0, 1, [])
    
    # Iterate M from smallest divisor to largest
    # To find the minimal max, we iterate through unique divisor values as potential M.
    # Actually, the max factor in the solution MUST be one of the divisors in the solution.
    # So we can just iterate through the sorted divisors.
    
    for M in divisors:
        subset = find_subset(target, divisors, M)
        if subset is not None:
            return subset
    
    # Fallback: just return [target] (which is > 12)
    return [target]

# EVOLVE-BLOCK-END
