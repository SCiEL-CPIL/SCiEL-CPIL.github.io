import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 12
    P = 27720  # LCM(1, ..., 12)
    MAX_L = 144
    
    # Precompute candidate masks for each residue r and each divisor i
    # masks[r][i] is a bitmask where bit j is set if (r + 1 + j) % i == 0 for j in 0..MAX_L-1
    # We only need to store this for the current r to save memory, or precompute all?
    # Precomputing all is 27720 * 12 ints. ~330k ints. ~2.6MB. Fine.
    # But calculating them takes time.
    
    # Let's calculate on the fly but optimize the inner loop.
    
    def compute_f_for_residue(r):
        """
        Compute f(12, r) using precomputed logic for this r.
        """
        # Build candidate masks for this r
        # cand_masks[i] for i in 1..12
        cand_masks = [0] * (N + 1)
        
        # For each i, determine which offsets j (0..143) satisfy (r+1+j) % i == 0
        for i in range(1, N + 1):
            mask = 0
            # (r + 1 + j) % i == 0  =>  j % i == (-r - 1) % i
            target_mod = (-r - 1) % i
            for j in range(MAX_L):
                if j % i == target_mod:
                    mask |= (1 << j)
            cand_masks[i] = mask
        
        # Order indices by number of candidates (popcount)
        # We need to do this for each L? No, the relative order might change as L increases?
        # Actually, popcount of (cand_masks[i] & ((1<<L)-1)) changes with L.
        # But for DFS, we just need a good heuristic.
        # We can just sort by total popcount in MAX_L, or dynamic.
        # Dynamic sorting is expensive. Let's use static order based on MAX_L candidates, 
        # or just fixed order 12, 11, 7, 5, ... (primes first, then composites)
        # Primes have fewer candidates.
        # 11: 144/11 ~ 13
        # 7: 144/7 ~ 20
        # 5: 144/5 ~ 28
        # 12: 144/12 = 12
        # Let's sort by i descending? No, by sparsity.
        
        # Let's try a fixed order: 11, 7, 5, 12, 10, 9, 8, 6, 4, 3, 2, 1
        # This is a heuristic. 11 and 7 are most constrained.
        
        order = [11, 7, 5, 12, 10, 9, 8, 6, 4, 3, 2, 1]
        
        # Check for L from 1 to MAX_L
        for L in range(1, MAX_L + 1):
            # Mask for valid slots 0..L-1
            valid_mask = (1 << L) - 1
            
            # Check if any i has no candidates in 0..L-1
            # If so, return L+1? No, if no matching for L, try L+1.
            # We need to check if matching exists.
            
            # Quick check: if any cand_masks[i] & valid_mask == 0, no matching.
            possible = True
            for i in order:
                if (cand_masks[i] & valid_mask) == 0:
                    possible = False
                    break
            
            if not possible:
                continue
                
            # Run DFS
            used = 0
            
            def dfs(idx):
                if idx == N:
                    return True
                i = order[idx]
                avail = cand_masks[i] & valid_mask & ~used
                while avail:
                    bit = avail & -avail
                    if dfs(idx + 1 | bit): 
                        # Note: dfs(idx+1) needs to know used. 
                        # Python closure captures 'used' by reference, so we must pass it or use nonlocal.
                        # Using nonlocal 'used' is tricky with recursion depth and state.
                        # Better to pass 'used' as argument.
                        pass
                    avail ^= bit
                return False
            
            # Rewrite DFS to pass used
            def dfs_helper(idx, used_mask):
                if idx == N:
                    return True
                i = order[idx]
                avail = cand_masks[i] & valid_mask & ~used_mask
                while avail:
                    bit = avail & -avail
                    if dfs_helper(idx + 1, used_mask | bit):
                        return True
                    avail ^= bit
                return False
            
            if dfs_helper(0, 0):
                return L
                
        return MAX_L + 1 # Should not happen if MAX_L is sufficient

    best_L = 0
    best_r = 0
    
    for r in range(P):
        if time.time() > deadline - 5:
            break
            
        L = compute_f_for_residue(r)
        if L > best_L:
            best_L = L
            best_r = r
            if best_L >= 144:
                break
                
    return int(best_r)
