import time
import sys

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
    
    best_m = 12
    best_L_found = 12
    
    # Phase 1: Quick greedy estimate for all m to identify candidates
    candidates = []
    for m in range(period):
        # Calculate minimal L required by individual constraints
        min_L = 0
        for i in range(1, n + 1):
            next_mult = ((m // i) + 1) * i
            dist = next_mult - m
            if dist > min_L:
                min_L = dist
        
        # Quick greedy check at L = min_L
        # Try to assign b_i greedily, starting from largest i
        assigned = set()
        greedy_ok = True
        for i in range(n, 0, -1):
            start_k = (m // i) + 1
            end_k = (m + min_L) // i
            found = False
            for k in range(start_k, end_k + 1):
                val = k * i
                if val not in assigned:
                    assigned.add(val)
                    found = True
                    break
            if not found:
                greedy_ok = False
                break
        
        # Score based on max distance for critical i values
        critical_max = 0
        for i in [9, 10, 11, 12]:
            next_mult = ((m // i) + 1) * i
            dist = next_mult - m
            if dist > critical_max:
                critical_max = dist
        
        # Higher critical_max and higher min_L suggests higher f(12,m)
        score = critical_max * 10 + min_L
        candidates.append((score, m))
    
    # Sort by score descending, take top candidates
    candidates.sort(reverse=True)
    
    # Take top 5000 candidates for full evaluation
    top_candidates = [m for _, m in candidates[:5000]]
    
    # Phase 2: Full binary search + matching on top candidates
    for m in top_candidates:
        if time.time() - start_time > max_time:
            break
            
        # Calculate minimal L required by individual constraints
        min_L = 0
        for i in range(1, n + 1):
            next_mult = ((m // i) + 1) * i
            dist = next_mult - m
            if dist > min_L:
                min_L = dist
        
        # Binary search for the minimum L such that a matching exists
        low = min_L
        high = max_L
        found_L = None
        
        while low <= high:
            mid = (low + high) // 2
            L = mid
            
            # Build adj masks for this L
            adj_masks = [0] * n
            
            for i in range(1, n + 1):
                start_k = (m // i) + 1
                end_k = (m + L) // i
                mask = 0
                for k in range(start_k, end_k + 1):
                    val = k * i
                    j = val - m - 1
                    if 0 <= j < L:
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
            L_val = max_L
            
        if L_val > best_L_found:
            best_L_found = L_val
            best_m = m

    return int(best_m)

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We place it inside the block as requested.
# EVOLVE-BLOCK-END
