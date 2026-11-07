import time

def construct():
    deadline = time.time() + 85  # Safety margin
    
    n = 12
    period = 27720  # lcm(1, ..., 12)
    max_L_bound = 200  # Upper bound for L, safe for n=12
    
    # Precompute f_values for one period
    f_values = [0] * period
    
    # Matching check using DFS with bitmasks
    # masks: list of 12 integers, each is a bitmask of valid indices in the interval
    def has_matching(masks):
        # Sort indices by number of candidates (popcount)
        idx_order = list(range(n))
        idx_order.sort(key=lambda i: masks[i].bit_count())
        
        # Quick check: if any mask is 0, no match
        for i in idx_order:
            if masks[i] == 0:
                return False
        
        used = 0
        
        # Recursive DFS
        # To avoid nested function definition overhead in the hot loop, 
        # we could use a global helper, but for this scale (27720 * ~20 * avg_L)
        # it might be fine. 
        # Total calls: 27720 * (sum of L checks).
        # If avg L is 10, that's 27720 * 10 = 277k matching checks.
        # Each check is small.
        
        def dfs(k):
            if k == n:
                return True
            
            i = idx_order[k]
            m = masks[i]
            # Iterate through set bits in m
            # We need to try bits not in 'used'
            available = m & ~used
            
            while available:
                # Get lowest set bit
                bit = available & -available
                available ^= bit
                
                # Try assigning this bit
                if dfs(k + 1, used | bit):
                    return True
            
            return False
        
        return dfs(0, 0)

    # We need to pass 'used' as argument to dfs to avoid nonlocal scope issues 
    # and potentially speed up closure access? 
    # Actually, passing 'used' as argument is cleaner for recursion.
    # Let's redefine has_matching to use a helper that takes used.
    
    def check_match(masks):
        idx_order = list(range(n))
        idx_order.sort(key=lambda i: masks[i].bit_count())
        
        for i in idx_order:
            if masks[i] == 0:
                return False
        
        # Use a stack or recursion. Recursion depth 12 is safe.
        # We'll use a nested function that captures idx_order and masks.
        # To optimize, we can pre-compute the list of bits for each mask?
        # No, bit manipulation is fast.
        
        def dfs(k, used):
            if k == n:
                return True
            i = idx_order[k]
            m = masks[i]
            available = m & ~used
            
            while available:
                bit = available & -available
                available ^= bit
                if dfs(k + 1, used | bit):
                    return True
            return False
        
        return dfs(0, 0)

    best_f = 0
    best_r = 0
    
    for r in range(period):
        if time.time() > deadline:
            break
        
        found = False
        # We want the LEAST L. 
        # To maximize f, we want the r where this least L is largest.
        # So we just compute f(r) exactly.
        
        # Optimization: We can stop checking L if we already know f(r) <= current best_f?
        # No, we need the exact f(r) to know if it's a new max.
        # But if we are just looking for the global max, we can prune?
        # If we find a match for L, f(r) = L.
        # If L < best_f, we don't need to continue? No, we found the min L.
        # So we just compute the min L for each r.
        
        f_r = 0
        for L in range(1, max_L_bound + 1):
            # Build masks for interval (r, r+L]
            # Numbers are r+1, ..., r+L
            # Index j corresponds to value r+1+j
            
            masks = [0] * n
            
            # Optimization: Instead of iterating j for each i, 
            # iterate j and check divisibility?
            # Or iterate i and find multiples?
            
            # For small L, iterating j is fine.
            # L is max ~144.
            # 12 * 144 operations per L.
            # Total ops per r: sum_{L=1}^{f(r)} 12*L.
            
            for j in range(L):
                v = r + 1 + j
                # Check divisibility by 1..12
                # We can precompute divisors for numbers up to r+max_L?
                # r is up to 27720. Max v ~ 27864.
                # Precomputing divisors for all numbers up to 30000 is cheap.
                
                # Let's just do modulo checks.
                # 12 checks per j.
                for i in range(1, 13):
                    if v % i == 0:
                        masks[i-1] |= (1 << j)
            
            if check_match(masks):
                f_r = L
                found = True
                break
        
        if not found:
            f_r = max_L_bound
        
        if f_r > best_f:
            best_f = f_r
            best_r = r
    
    # The optimal m is the smallest m in [0, 1000000] with m % period == best_r
    # Since best_r < period, m = best_r is the smallest such m.
    # We must ensure m <= 1000000.
    # 27720 < 1000000, so best_r is valid.
    
    return int(best_r)
