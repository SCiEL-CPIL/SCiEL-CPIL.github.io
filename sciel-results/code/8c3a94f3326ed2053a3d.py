def _can_add_with_dp(dp, current_set, cand, n):
    if n == 0:
        return True
    
    # 1. cand is not mean of any subset of current_set (size >= 2)
    for k in range(2, n + 1):
        target = k * cand
        if (dp[k] >> target) & 1:
            return False
            
    # 2. No a in current_set is mean of subset of (current_set U {cand}) including cand.
    # For each k in 1..n:
    #   Compute mask of targets: t = a*(k+1) - cand for a in current_set
    #   If dp[k] & mask != 0, return False
    
    for k in range(1, n + 1):
        mask = 0
        factor = k + 1
        for a in current_set:
            t = a * factor - cand
            if t >= 0:
                mask |= (1 << t)
        if dp[k] & mask:
            return False
            
    return True
