import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj_masks[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_independent[rest] and (adj_masks[v] & rest) == 0:
            is_independent[mask] = True
    
    # Build list of all valid non-empty parts (masks that are clique or independent)
    valid_parts = []
    for mask in range(1, size):
        if is_clique[mask] or is_independent[mask]:
            valid_parts.append(mask)
    
    # Sort valid parts by size descending (larger parts first for better pruning)
    valid_parts.sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Try to find minimal number of parts using iterative deepening DFS
    # Try k = 1, 2, 3, ... until we find a valid cover
    best_parts = None
    
    def dfs(remaining_mask, parts_used, max_parts):
        """Try to cover remaining_mask with at most max_parts - parts_used parts."""
        if time.time() > deadline:
            return False
        if remaining_mask == 0:
            return True
        if parts_used == max_parts:
            return False
        
        # Lower bound: we need at least ceil(popcount(remaining) / max_part_size) more parts
        # But simpler: if we've used all allowed parts, fail
        if parts_used >= max_parts:
            return False
        
        # Try each valid part that is a subset of remaining_mask
        # Optimization: only try parts that contain the smallest vertex in remaining_mask
        # to avoid symmetric solutions
        min_vertex = (remaining_mask & -remaining_mask).bit_length() - 1
        
        for part_mask in valid_parts:
            # Part must be subset of remaining
            if part_mask & ~remaining_mask:
                continue
            # Part must contain the minimum vertex (to break symmetry)
            if not (part_mask & (1 << min_vertex)):
                continue
            # Part must be non-empty (guaranteed by valid_parts)
            
            if dfs(remaining_mask ^ part_mask, parts_used + 1, max_parts):
                return True
        
        return False
    
    # Try k = 1, 2, 3, ...
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Reset and try
        # We need to track which parts we used, so use a different approach
        # Use iterative deepening with part tracking
        parts_found = []
        
        def dfs_with_tracking(remaining_mask, parts_used, max_parts, current_parts):
            if time.time() > deadline:
                return False
            if remaining_mask == 0:
                if parts_used <= max_parts:
                    current_parts.extend(parts_found)
                    return True
                return False
            if parts_used >= max_parts:
                return False
            
            min_vertex = (remaining_mask & -remaining_mask).bit_length() - 1
            
            for part_mask in valid_parts:
                if part_mask & ~remaining_mask:
                    continue
                if not (part_mask & (1 << min_vertex)):
                    continue
                
                # Convert mask to list
                part_list = []
                for i in range(n):
                    if part_mask & (1 << i):
                        part_list.append(i)
                
                current_parts.append(part_list)
                if dfs_with_tracking(remaining_mask ^ part_mask, parts_used + 1, max_parts, current_parts):
                    return True
                current_parts.pop()
            
            return False
        
        if dfs_with_tracking(full_mask, 0, k, []):
            best_parts = list(current_parts)
            break
    
    # Fallback: if DFS didn't find anything or timed out, use DP
    if best_parts is None or time.time() > deadline:
        # Run DP as fallback
        dp = [float('inf')] * size
        parent = [-1] * size
        dp[0] = 0
        
        for mask in range(1, size):
            if time.time() > deadline:
                break
            s = mask
            while s:
                if is_clique[s] or is_independent[s]:
                    prev = mask ^ s
                    if dp[prev] + 1 < dp[mask]:
                        dp[mask] = dp[prev] + 1
                        parent[mask] = s
                s = (s - 1) & mask
        
        if dp[full_mask] != float('inf') and time.time() < deadline:
            parts = []
            mask = full_mask
            while mask:
                s = parent[mask]
                if s == -1:
                    break
                part = []
                for i in range(n):
                    if s & (1 << i):
                        part.append(i)
                parts.append(part)
                mask ^= s
            best_parts = parts
    
    # Final fallback: greedy
    if best_parts is None:
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        best_parts = parts
    
    # Ensure all parts are lists of ints
    result = []
    for part in best_parts:
        result.append([int(v) for v in part])
    
    return result
