def construct():
    import math
    
    def sqf(n):
        if n in sqf_cache:
            return sqf_cache[n]
        orig = n
        sf = 1
        p = 2
        while p * p <= n:
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                sf *= p
            p += 1
        if n > 1:
            sf *= n
        sqf_cache[orig] = sf
        return sf

    sqf_cache = {}
    for i in range(1, 100):
        sqf(i)
        
    target = 210
    
    # reachable: dict sqf_val -> list of numbers
    # We want to minimize the max number.
    # We iterate K from 211 to 10210.
    # At step K, we add K to the pool.
    # We update reachable states by combining with K.
    
    reachable = {1: []}
    
    for K in range(211, 10211):
        sK = sqf(K)
        # New states formed by adding K to existing states
        # We must be careful not to use K more than once.
        # So we iterate over the *old* reachable states.
        
        new_states = {}
        for curr_s, path in reachable.items():
            # Combine curr_s and sK
            prod = curr_s * sK
            new_s = sqf(prod)
            
            # If this new_s is not already in reachable, or if the new path is "better"?
            # We don't need "better", just *a* path.
            # But we want to minimize max. Since we are iterating K, the first time we find target is the min max.
            # However, we might find target using a path that doesn't include K?
            # No, if target was in reachable, we would have returned in a previous iteration.
            # So we only need to check if new_s == target.
            
            if new_s not in reachable and new_s not in new_states:
                # Only add if not already reachable via shorter/smaller path?
                # Actually, we just want to know if it's reachable.
                # Storing the path is for returning.
                new_states[new_s] = path + [K]
                
        # Update reachable
        for s, p in new_states.items():
            if s not in reachable:
                reachable[s] = p
                
        if target in reachable:
            sol = reachable[target]
            # Verify
            prod = 210
            for x in sol:
                prod *= x
            root = math.isqrt(prod)
            if root * root == prod:
                return [int(x) for x in sol]
                
    return [224, 240]
