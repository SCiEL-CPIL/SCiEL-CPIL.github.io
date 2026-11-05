def construct():
    from math import isqrt
    import numpy as np

    # Generate all primes <= 10000
    primes = []
    for p in range(2, 10001):
        if all(p % d for d in range(2, isqrt(p) + 1)):
            primes.append(p)
    
    n = len(primes)
    if n < 2:
        return primes

    max_gap = primes[-1] - primes[0]
    
    # dp[i][g] is not stored for all i, only current i.
    # best[i][g] = max length of subseq ending at i with last gap <= g
    # best_idx[i][g] = the gap h <= g that achieves best[i][g]
    # parent_i[i][g] = the index j of the previous prime
    # parent_g[i][g] = the gap h_prev used in the subseq ending at j
    
    # Initialize with base case: length 1, gap 0 (virtual)
    # best[i][g] = 1 for all g
    # best_idx[i][g] = 0 for all g (indicating no previous gap, or gap 0)
    # Note: gap 0 is not a real gap between distinct primes, but serves as identity for non-decreasing check.
    
    best = np.ones((n, max_gap + 1), dtype=np.int32)
    best_idx = np.zeros((n, max_gap + 1), dtype=np.int32)
    parent_i = np.full((n, max_gap + 1), -1, dtype=np.int32)
    parent_g = np.zeros((n, max_gap + 1), dtype=np.int32)
    
    # dp for current i
    # We need to reset dp for each i
    # Actually, we can just use a list or array and set to 0 before each i
    
    for i in range(1, n):
        pi = primes[i]
        dp = np.zeros(max_gap + 1, dtype=np.int32)
        
        for j in range(i):
            g = pi - primes[j]
            if g > max_gap:
                continue
            
            # Query best[j][g]
            len_val = best[j][g]
            idx_val = best_idx[j][g]
            
            cand_len = len_val + 1
            
            if cand_len > dp[g]:
                dp[g] = cand_len
                parent_i[i][g] = j
                parent_g[i][g] = idx_val
        
        # Compute best[i] and best_idx[i] from dp
        # best[i][g] = max(1, max(dp[0..g]))
        # best_idx[i][g] = argmax(dp[0..g]) if max > 1 else 0
        
        curr_max = 1
        curr_idx = 0
        for g in range(max_gap + 1):
            if dp[g] > curr_max:
                curr_max = dp[g]
                curr_idx = g
            best[i][g] = curr_max
            best_idx[i][g] = curr_idx

    # Find global max
    max_len = 0
    best_i = -1
    best_g = -1
    
    for i in range(n):
        # The max length ending at i is best[i][max_gap]
        # But we need the specific gap that achieved it to backtrack?
        # No, best[i][max_gap] gives the max length.
        # But we need to know which gap g to start backtracking from.
        # If best[i][max_gap] = L, there is some g such that dp[i][g] = L.
        # Actually, best[i][g] is non-decreasing in g.
        # The value L is achieved for all g >= g_min where dp[i][g_min]=L.
        # We can just pick the largest g? Or any g where best[i][g] == L?
        # Let's find the g that maximizes dp[i][g].
        # Or simply, iterate through all g?
        # We stored parent_i[i][g] and parent_g[i][g].
        # We need to find a g such that dp[i][g] is the max.
        # Actually, we can just check best[i][max_gap].
        # If best[i][max_gap] > max_len:
        #    max_len = best[i][max_gap]
        #    best_i = i
        #    # Now find the g that achieved this.
        #    # We can find the first g where best[i][g] == max_len?
        #    # Or the last one?
        #    # It doesn't matter for validity, as long as parent pointers exist.
        #    # Let's find the g such that dp[i][g] == max_len.
        #    # We can just scan dp[i]? No, dp[i] is local.
        #    # We can scan best[i] and best_idx[i].
        #    # Actually, if best[i][g] = L, then best_idx[i][g] is the gap h.
        #    # We can just use best_idx[i][max_gap] as the starting gap?
        #    # Yes, because best_idx[i][max_gap] points to the gap h <= max_gap that achieved the max length.
        #    # So we set best_g = best_idx[i][max_gap].
        
        if best[i][max_gap] > max_len:
            max_len = best[i][max_gap]
            best_i = i
            best_g = best_idx[i][max_gap]

    # Reconstruct
    if best_i == -1:
        return primes[:2]
        
    seq = []
    cur_i = best_i
    cur_g = best_g
    
    while cur_i != -1:
        seq.append(primes[cur_i])
        if cur_i == 0:
            break
        j = parent_i[cur_i][cur_g]
        prev_g = parent_g[cur_i][cur_g]
        cur_i = j
        cur_g = prev_g
        
    seq.reverse()
    return seq
