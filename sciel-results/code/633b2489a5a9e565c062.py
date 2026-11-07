import time

def has_perfect_matching_bitmask(left_size, adj_masks, right_size):
    """
    left_size: number of left nodes (12)
    adj_masks: list of integers, bit j set if right node j is connected to left node i
    right_size: number of right nodes
    Returns True if perfect matching exists.
    Uses DFS (Kuhn's algorithm) with bitmasks, ordered by degree.
    """
    # Early exit: if any left node has no options
    for i in range(left_size):
        if adj_masks[i] == 0:
            return False
    
    match_to_right = [-1] * right_size
    
    # Order left nodes by degree (fewest options first) for better pruning
    degrees = [(adj_masks[i].bit_count(), i) for i in range(left_size)]
    degrees.sort()
    order = [i for _, i in degrees]
    
    def dfs(u, seen_mask):
        avail = adj_masks[u] & ~seen_mask
        while avail:
            v = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            if match_to_right[v] == -1 or dfs(match_to_right[v], seen_mask | (1 << v)):
                match_to_right[v] = u
                return True
        return False
    
    for u in order:
        if not dfs(u, 0):
            return False
    return True

def construct():
    start_time = time.time()
    max_time = 85.0 # Leave 5s margin
    
    n = 12
    max_L = 144
    period = 27720 # LCM(1..12)
    
    best_m = 0
    best_L_found = 0
    
    # We will scan the entire period.
    # The cost per m is ~log2(144) * (matching_cost).
    # matching_cost is very low for n=12.
    # Total time should be well within limits.
    
    for m in range(period):
        if time.time() - start_time > max_time:
            break
            
        # Calculate minimal L required by individual constraints
        # L must be at least the distance to the first multiple of any i
        min_L = 0
        for i in range(1, n + 1):
            # Smallest multiple of i strictly greater than m
            # (m // i) * i <= m
            # ((m // i) + 1) * i > m
            next_mult = ((m // i) + 1) * i
            dist = next_mult - m
            if dist > min_L:
                min_L = dist
        
        # Binary search for the minimum L such that a matching exists
        # L is in [min_L, max_L]
        low = min_L
        high = max_L
        found_L = None
        
        while low <= high:
            mid = (low + high) // 2
            L = mid
            
            # Build adj masks for this L
            # Right nodes are indices 0 to L-1, representing values m+1 to m+L
            adj_masks = [0] * n
            
            for i in range(1, n + 1):
                start_k = (m // i) + 1
                end_k = (m + L) // i
                mask = 0
                # Iterate through multiples of i in (m, m+L]
                for k in range(start_k, end_k + 1):
                    val = k * i
                    j = val - m - 1 # Index in the right side (0-based)
                    # j is guaranteed to be in [0, L-1] because val <= m+L and val > m
                    mask |= (1 << j)
                adj_masks[i-1] = mask
            
            if has_perfect_matching_bitmask(n, adj_masks, L):
                found_L = L
                high = mid - 1
            else:
                low = mid + 1
        
        if found_L is not None:
            L_val = found_L
        else:
            # If no matching found up to max_L, cap it (should not happen for max_L=144 
            # as 144 is a safe upper bound for n=12)
            L_val = max_L
            
        if L_val > best_L_found:
            best_L_found = L_val
            best_m = m

    return int(best_m)

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END
