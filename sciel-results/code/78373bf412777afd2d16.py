def construct():
    import time
    start = time.time()
    DEADLINE = 85.0
    n = 12
    PERIOD = 27720  # lcm(1..12)
    MAXL = 144
    MINL = 12

    def can(m, L):
        cands = []
        for i in range(1, n+1):
            r = (-(m+1)) % i
            mask = 0
            for p in range(r, L, i):
                mask |= 1 << p
            cands.append(mask)
        
        # Check if any variable has no candidates
        for i in range(n):
            if cands[i] == 0:
                return False
        
        # Memoization on (used) - we track which positions are used
        # Use a set-based approach with dynamic MRV
        memo = {}
        
        def rec(used):
            if used.bit_count() == n:
                return True
            if used in memo:
                return memo[used]
            
            # Dynamic MRV: pick the variable with fewest available candidates
            best_i = -1
            best_count = L + 1
            for i in range(n):
                avail = cands[i] & ~used
                cnt = avail.bit_count()
                if cnt == 0:
                    memo[used] = False
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_i = i
                    if cnt == 1:
                        break
            
            i = best_i
            mask = cands[i] & ~used
            result = False
            while mask:
                lsb = mask & (-mask)
                if rec(used | lsb):
                    result = True
                    break
                mask &= mask - 1
            memo[used] = result
            return result
        
        return rec(0)

    def f_of(m):
        lo, hi = MINL, MAXL
        if not can(m, hi):
            return -1
        while lo < hi:
            mid = (lo+hi)//2
            if can(m, mid):
                hi = mid
            else:
                lo = mid+1
        return lo

    best_m = 0
    best_f = -1
    for m in range(PERIOD):
        if time.time() - start > DEADLINE:
            break
        fv = f_of(m)
        if fv > best_f:
            best_f = fv
            best_m = m
    return int(best_m)
