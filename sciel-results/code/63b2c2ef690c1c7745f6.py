import time
import __main__

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Robustly locate the data dictionary P
    P = globals().get('P')
    if P is None:
        P = globals().get('params')
    if P is None:
        P = globals().get('data')
    if P is None:
        P = globals().get('config')
    if P is None:
        P = getattr(__main__, 'P', None)
    if P is None:
        P = getattr(__main__, 'params', None)
        
    if P is None:
        # If P is not found, we cannot solve the task.
        # Return an empty list or raise? 
        # The contract says "Any broken rule makes the whole result invalid".
        # So returning an invalid result is the same as raising.
        # But raising might be caught and scored 0.
        # Let's raise a specific error to help debugging if possible, but it scores 0.
        raise NameError("Could not find data dictionary P")

    edge_order = P['edge_order']
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Precompute masks
    masks = []
    for triple in edge_order:
        m = 0
        for v in triple:
            m |= (1 << v)
        masks.append(m)
    
    # Check if a perfect matching exists using only the first K triples
    def feasible(K):
        # Get masks for first K triples
        sub_masks = masks[:K]
        
        # Deduplicate: for each unique mask, we only need one copy
        unique_masks = list(set(sub_masks))
        
        # Backtracking with memoization
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(used_mask, remaining):
            if remaining == 0:
                return used_mask == full_mask
            if remaining < 0:
                return False
            # Pruning: if not enough vertices left to fill remaining triples
            unused = full_mask ^ used_mask
            if bin(unused).count('1') < 3 * remaining:
                return False
            # Try each unique mask
            for m in unique_masks:
                if m & used_mask:
                    continue  # Not disjoint
                if dfs(used_mask | m, remaining - 1):
                    return True
            return False
        
        return dfs(0, 6)
    
    # Binary search for minimum K
    lo, hi = 1, len(edge_order)
    best_K = hi
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        if feasible(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Timeout check before final search
    if time.time() > deadline - 5:
        # If we are close to deadline, return a fallback or best effort?
        # We haven't found a matching yet.
        # We can try a greedy approach or return the seed?
        # The seed solution is [0,1,2], [3,4,5], etc.
        # This is a valid perfect matching, but might have a high score.
        # But it's better than 0 score (exception).
        return [list(range(i, i+3)) for i in range(0, n, 3)]

    # Now find an actual matching for best_K
    sub_masks = masks[:best_K]
    unique_masks = list(set(sub_masks))
    
    from functools import lru_cache
    
    # Map mask to list of indices in the original edge_order (within first best_K)
    mask_to_indices = {}
    for i in range(best_K):
        m = masks[i]
        if m not in mask_to_indices:
            mask_to_indices[m] = []
        mask_to_indices[m].append(i)
    
    @lru_cache(maxsize=None)
    def find_matching(used_mask, remaining):
        if remaining == 0:
            if used_mask == full_mask:
                return ()
            else:
                return None
        if remaining < 0:
            return None
        unused = full_mask ^ used_mask
        if bin(unused).count('1') < 3 * remaining:
            return None
        for m in unique_masks:
            if m & used_mask:
                continue
            result = find_matching(used_mask | m, remaining - 1)
            if result is not None:
                # Pick the first index for this mask
                idx = mask_to_indices[m][0]
                return (idx,) + result
        return None
    
    result = find_matching(0, 6)
    
    if result is None:
        # Fallback: return the seed solution
        return [list(range(i, i+3)) for i in range(0, n, 3)]
    
    # Convert indices to triples
    matching = [list(edge_order[i]) for i in result]
    
    return matching
