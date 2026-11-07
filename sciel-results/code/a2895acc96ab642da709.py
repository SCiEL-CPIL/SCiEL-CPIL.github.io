import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144
    
    # Precompute bitmasks for valid offsets
    # mask[i][rem] is an integer where bit d is 1 if d % i == rem
    # We only care about d in 1..max_L
    valid_masks = {}
    for i in range(1, n + 1):
        valid_masks[i] = [0] * i
        for d in range(1, max_L + 1):
            rem = d % i
            valid_masks[i][rem] |= (1 << d)
    
    # Mask to limit d to 1..L
    # We will compute this on the fly or precompute for all L?
    # Precomputing for all L is 144 * 12 * 12 integers, which is small.
    # But (1 << (L+1)) - 2 is fast enough.
    
    def has_perfect_matching(m, L):
        """
        Check if (m, m+L] has a perfect matching for distinct multiples.
        Left nodes: 1..12
        Right nodes: 1..L
        Edge (i, d) exists if (m+d) % i == 0, i.e., d % i == (-m) % i.
        """
        if L < n:
            return False
        
        # Build adjacency bitmasks for each left node i
        # adj[i-1] will be the bitmask of valid d's for node i
        adj = [0] * n
        
        for i in range(1, n + 1):
            rem = (-m) % i
            # Get precomputed mask for this residue
            mask = valid_masks[i][rem]
            # Mask out d > L
            # Bits 0..L-1 are 0, bits 1..L are 1, bits > L are 0
            # (1 << (L+1)) - 2 gives bits 1..L set to 1
            limit_mask = (1 << (L + 1)) - 2
            adj[i - 1] = mask & limit_mask
            
            # Early exit if no options
            if adj[i - 1] == 0:
                return False
        
        # Bipartite Matching using DFS with bitmasks
        # matchR[d] = i (0-indexed left node) that d is matched to
        # We use a list of size L+1, index 0 unused
        matchR = [-1] * (L + 1)
        
        # To speed up DFS, we can order left nodes by number of options
        # But with bitmasks, iterating through set bits is fast.
        # Let's stick to a simple DFS but optimized.
        
        def dfs(i, seen):
            # i is 0-indexed left node
            # We want to find an augmenting path for left node i
            # seen is a bitmask of right nodes visited in this DFS call
            # We iterate through set bits in adj[i]
            avail = adj[i] & ~seen
            while avail:
                # Get lowest set bit
                lsb = avail & -avail
                d = lsb.bit_length() - 1
                avail ^= lsb
                seen |= lsb
                
                if matchR[d] == -1 or dfs(matchR[d], seen):
                    matchR[d] = i
                    return True
            return False
        
        # Order left nodes by number of available options (popcount)
        # This helps DFS find matches faster
        order = sorted(range(n), key=lambda idx: bin(adj[idx]).count('1'))
        
        for orig_idx in order:
            seen = 0
            if not dfs(orig_idx, seen):
                return False
        return True

    def find_min_L(m):
        """Binary search for minimum L for a given m."""
        lo, hi = 1, max_L
        min_L_for_m = max_L
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                min_L_for_m = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return min_L_for_m

    best_m = 0
    best_L = 0
    
    # Scan all m in the period
    # 27720 iterations.
    # Each find_min_L does ~7 matching checks (binary search).
    # Each matching check is very fast with bitmasks.
    
    for m in range(period):
        if time.time() - start_time > deadline:
            break
        
        min_L_for_m = find_min_L(m)
        
        if min_L_for_m > best_L:
            best_L = min_L_for_m
            best_m = m
        
        # Early stopping if we hit the theoretical max
        if best_L >= max_L:
            break

    if best_m < 0 or best_m > 1000000:
        best_m = 0
    
    return int(best_m)
