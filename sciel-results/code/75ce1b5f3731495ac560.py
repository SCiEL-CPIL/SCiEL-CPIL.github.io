def construct():
    from fractions import Fraction
    import bisect

    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [Fraction(1, d) for d in denominators]
    
    # Split into two halves
    mid = n // 2
    left_vals = vals[:mid]
    right_vals = vals[mid:]
    
    def generate_valid_subsets(values):
        """Generates list of (sum, mask) for subsets with sum < 1."""
        results = []
        n_local = len(values)
        
        def dfs(idx, curr_sum, curr_mask):
            if idx == n_local:
                results.append((curr_sum, curr_mask))
                return
            
            # Exclude
            dfs(idx + 1, curr_sum, curr_mask)
            
            # Include
            new_sum = curr_sum + values[idx]
            if new_sum < 1:
                dfs(idx + 1, new_sum, curr_mask | (1 << idx))
        
        dfs(0, Fraction(0), 0)
        return results

    # Generate subsets
    left_subsets = generate_valid_subsets(left_vals)
    right_subsets = generate_valid_subsets(right_vals)
    
    # Sort right subsets by sum
    right_subsets.sort(key=lambda x: x[0])
    right_sums = [x[0] for x in right_subsets]
    
    best_sum = Fraction(-1)
    best_mask_left = 0
    best_mask_right = 0
    
    # For each left subset, find best right subset
    for s_l, m_l in left_subsets:
        # We want max s_r such that s_l + s_r < 1 => s_r < 1 - s_l
        target = 1 - s_l
        # Find rightmost index in right_sums where value < target
        # bisect_left returns first index >= target. We want the one before.
        idx = bisect.bisect_left(right_sums, target) - 1
        
        if idx >= 0:
            s_r, m_r = right_subsets[idx]
            total_sum = s_l + s_r
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask_left = m_l
                best_mask_right = m_r
                
    # Construct final indices
    indices = []
    # Add indices from left
    for i in range(mid):
        if best_mask_left & (1 << i):
            indices.append(i)
            
    # Add indices from right
    for i in range(len(right_vals)):
        if best_mask_right & (1 << i):
            indices.append(mid + i)
            
    return indices
