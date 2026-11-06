def construct():
    max_target = 128
    # Generate smooth numbers
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)
            
    n_smooth = len(smooth)
    
    # Precompute conflict matrix
    # conflict[i][j] is True if smooth[i] and smooth[j] cannot be in the same antichain
    conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if a % b == 0 or b % a == 0:
                conflict[i][j] = True
                conflict[j][i] = True
                
    # min_size_for_sum[s] = minimum number of terms
    # best_set_for_sum[s] = the set of indices (or values)
    min_size_for_sum = [float('inf')] * (max_target + 1)
    best_set_for_sum = [None] * (max_target + 1)
    
    # Enumerate independent sets (antichains)
    # We use a recursive DFS.
    # State: current index in smooth, current sum, current size, current set of indices
    # To avoid generating permutations, we only pick indices > last_picked_index.
    
    def dfs(start_idx, current_sum, current_size, current_set):
        # Update best if current sum is valid and size is smaller
        if current_sum <= max_target:
            if current_size > 0: # Must have at least 1 term? 
                # The problem says "1..30 distinct positive ... terms". 
                # Target 1 needs 1 term. 
                # Can a target be represented by 0 terms? No, sum would be 0.
                # So current_size must be >= 1 for sum > 0.
                if current_size < min_size_for_sum[current_sum]:
                    min_size_for_sum[current_sum] = current_size
                    best_set_for_sum[current_sum] = list(current_set)
        
        # Pruning: if current_sum > max_target, return
        if current_sum > max_target:
            return
            
        # Try to add more elements
        # We can pick any index j > start_idx such that it doesn't conflict with any element in current_set
        # To optimize, we can precompute for each index i, the list of indices j > i that conflict with i.
        # But checking against current_set is O(k).
        
        # Optimization: 
        # If we pick an element, we must ensure it doesn't conflict with any in current_set.
        # We can maintain a bitmask of "forbidden" indices?
        # Since n_smooth is ~37, we can use an integer bitmask.
        
        pass # Placeholder for the actual logic below

    # Let's implement the DFS with bitmask for forbidden nodes to speed up conflict checking.
    # forbidden_mask bit j is 1 if node j cannot be picked.
    
    # Precompute conflict masks for each node i: 
    # conflict_mask[i] has bit j set if node i conflicts with node j.
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if conflict[i][j]:
                conflict_mask[i] |= (1 << j)
                
    # Also, we need to respect the "start_idx" ordering to avoid permutations.
    # We can just iterate j from start_idx to n_smooth.
    # But we must check if j is in forbidden_mask.
    
    def dfs(start_idx, current_sum, current_size, current_set, forbidden_mask):
        if current_sum > max_target:
            return
        
        # Record solution
        if current_size > 0:
            if current_size < min_size_for_sum[current_sum]:
                min_size_for_sum[current_sum] = current_size
                best_set_for_sum[current_sum] = list(current_set)
                
        # Try adding more nodes
        # We can pick any node j >= start_idx
        # such that bit j is NOT set in forbidden_mask
        # and j is not already in current_set (implied by j >= start_idx and we only add once)
        
        # To be efficient, we can find the next available j.
        # However, simply iterating j from start_idx to n_smooth is fine given the small N.
        
        for j in range(start_idx, n_smooth):
            # Check if j is forbidden
            if (forbidden_mask >> j) & 1:
                continue
                
            # Check if adding j exceeds sum
            new_sum = current_sum + smooth[j]
            if new_sum > max_target:
                # Since smooth is sorted, if smooth[j] is too big, subsequent ones are too?
                # Yes, smooth is sorted ascending.
                # But wait, we are iterating j upwards. If smooth[j] makes sum too big, 
                # larger j will also make sum too big.
                # So we can break?
                # No, because we might skip some intermediate j's? 
                # No, we are iterating sequentially. If smooth[j] is too big, smooth[j+1] is also too big.
                # So we can break.
                break
                
            # New forbidden mask: old forbidden OR conflict_mask[j]
            new_forbidden_mask = forbidden_mask | conflict_mask[j]
            
            # Recurse
            # Next start index is j + 1 to ensure distinct and no permutations
            current_set.append(j)
            dfs(j + 1, new_sum, current_size + 1, current_set, new_forbidden_mask)
            current_set.pop()
            
    dfs(0, 0, 0, [], 0)
    
    results = []
    for i in range(1, max_target + 1):
        if min_size_for_sum[i] == float('inf'):
            # Should not happen
            raise ValueError(f"No representation found for {i}")
        # Convert indices back to values
        idxs = best_set_for_sum[i]
        vals = [smooth[idx] for idx in idxs]
        results.append(vals)
        
    return results
