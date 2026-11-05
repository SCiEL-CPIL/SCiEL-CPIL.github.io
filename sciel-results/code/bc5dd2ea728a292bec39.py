"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial

    target = factorial(12)  # 479001600

    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
        
    # Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Map divisor to index for quick lookup? Not needed for DP set.
    # But we need to ensure we only use divisors of target.
    # The DP will only generate products that are divisors of target if we check `target % prod == 0`.
    
    def find_decomposition(max_val):
        """
        Returns a list of distinct divisors <= max_val whose product is target, or None.
        Uses DP to find if possible, then backtracks to find the set.
        """
        candidates = [d for d in valid_divs if d <= max_val]
        if not candidates:
            return None
        
        # dp[product] = list of factors used to reach this product?
        # Or just a set of reachable products?
        # To backtrack, we need to know which factor was added.
        # Let's use a dict: product -> (prev_product, factor_added)
        # But multiple paths might lead to same product. We just need one.
        
        # Start with product 1
        reachable = {1: None} # product -> (prev_prod, factor)
        # We iterate through candidates. To ensure distinctness, we process each candidate once.
        # This is 0/1 knapsack style.
        
        # However, standard 0/1 knapsack iterates items and updates DP.
        # If we just use a set `current_reachable`, we lose the path.
        # We can store `parent[product] = (prev_product, factor)`
        
        parent = {1: (0, 0)} # product -> (prev_product, factor). 0 indicates start.
        
        # We need to be careful: we can only use each candidate once.
        # So we iterate candidates in order.
        # For each candidate c, we look at all currently reachable products p.
        # If p * c is a divisor of target, we mark p*c as reachable with parent (p, c).
        # We must not use c again. Since we process c once, and update a *new* set or update in place?
        # If we update in place, we might use c multiple times (unbounded knapsack).
        # So we must use a snapshot or iterate backwards?
        # Products are not indices.
        # We can maintain a list of reachable products.
        # For each c, we create a new list of additions.
        
        reachable_set = {1}
        # We need to track the path.
        # Let's store for each reachable product, the last factor added and the previous product.
        # But if a product is reached via multiple paths, we just keep one.
        # However, we must ensure we don't reuse factors.
        # The "0/1" nature is handled by processing each candidate exactly once.
        
        # Algorithm:
        # reachable = {1}
        # parent = {1: None}
        # for c in candidates:
        #    new_reachable = set()
        #    for p in reachable:
        #        prod = p * c
        #        if target % prod == 0:
        #            if prod not in parent: # Only record first time? Or any time?
        #                # If we record first time, is it safe?
        #                # Yes, because we just need *a* valid set.
        #                # But wait, if we reach `prod` earlier, we might have used a different set of factors.
        #                # Does it matter? No, as long as the factors are distinct and <= max_val.
        #                # The distinctness is guaranteed by the 0/1 iteration.
        #                parent[prod] = (p, c)
        #                new_reachable.add(prod)
        #    reachable.update(new_reachable)
        #    if target in reachable:
        #         # Backtrack
        #         res = []
        #         curr = target
        #         while parent[curr] is not None:
        #             prev, c = parent[curr]
        #             res.append(c)
        #             curr = prev
        #         return res
        # return None

        reachable = {1}
        parent = {1: None}
        
        for c in candidates:
            # We must iterate over the *current* state of reachable, not the updated one,
            # to avoid using c multiple times.
            current_reachable = list(reachable)
            for p in current_reachable:
                prod = p * c
                if target % prod == 0:
                    if prod not in parent:
                        parent[prod] = (p, c)
                        reachable.add(prod)
                        if prod == target:
                            # Backtrack
                            res = []
                            curr = target
                            while parent[curr] is not None:
                                prev, factor = parent[curr]
                                res.append(factor)
                                curr = prev
                            return res
        return None

    # Binary search on the maximum factor M
    # We want the smallest M such that find_decomposition(M) is not None.
    
    lo = 0
    hi = len(valid_divs) - 1
    best_max = valid_divs[-1]
    best_set = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        res = find_decomposition(valid_divs[mid])
        if res:
            best_max = valid_divs[mid]
            best_set = res
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_set is None:
        return [target]
        
    return best_set

# EVOLVE-BLOCK-END
