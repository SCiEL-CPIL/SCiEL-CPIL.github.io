"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200, 
    maximizing cardinality.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    # Start with all numbers
    selected = set(range(1, 101))
    
    def find_subset_summing_to_target(target, elements):
        """
        Finds a subset of elements summing to target.
        Returns the subset as a list, or None if not possible.
        Uses DP with backtracking.
        """
        # elements is a list
        n = len(elements)
        if n == 0:
            return None
        
        # DP table: dp[i][j] = True if sum j is possible using first i elements
        # To save memory and time, we use 1D array for reachability and parent pointers for one path
        # But we need to reconstruct the path, so we need to know which elements were used.
        # Standard 1D DP overwrites state. 
        # Given target 200 is small, we can use a 2D DP or store predecessors.
        # Let's use a dictionary or list of bitsets if possible, or just 2D bool list.
        # 100 elements, 200 sum. 100x200 = 20k bools. Fine.
        
        # dp[i][j] indicates if sum j can be formed by a subset of elements[0..i-1]
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = True
        
        # Fill DP table
        for i in range(1, n + 1):
            val = elements[i-1]
            for j in range(target + 1):
                if dp[i-1][j]:
                    dp[i][j] = True
                    if val <= j and dp[i-1][j-val]:
                         # This doesn't track the "last added" correctly for backtracking if we just set True.
                         # We need to know if dp[i][j] is true BECAUSE we took val.
                         pass
        
        # Better approach for backtracking:
        # dp[i][j] = True if sum j is possible with elements[0..i-1]
        # parent[i][j] = (parent_idx, was_val_used)
        
        # Let's just use a simple recursive search with memoization if DP is complex to backtrack.
        # Or, since target is small, we can just find ONE subset.
        
        # Simple DFS with pruning?
        # Elements sorted descending?
        
        # Let's stick to a clean DP backtracking implementation.
        # dp[i][j] is boolean.
        # prev[i][j] stores the index of the element added to reach state (i-1, j-val) if val was used,
        # or None if it was not used (inherited from i-1, j).
        
        # Actually, simpler:
        # We just need ANY subset.
        # We can use a set of reachable sums and store the subset that formed them?
        # Memory might be high, but 200 sums is small.
        # reachable_sums: {sum_value: subset_list}
        
        reachable = {0: []}
        
        for val in elements:
            new_reachable = {}
            # Iterate over current reachable sums
            # We must not use 'val' more than once, so we iterate over the snapshot of 'reachable'
            for s, subset in list(reachable.items()):
                new_sum = s + val
                if new_sum <= target:
                    if new_sum not in reachable and new_sum not in new_reachable:
                        new_reachable[new_sum] = subset + [val]
            
            # Merge
            for k, v in new_reachable.items():
                if k not in reachable:
                    reachable[k] = v
            # Optimization: If target is reached, we can return early?
            # But we need to finish processing current element? No, if we found a valid subset, we can return it immediately.
            if target in reachable:
                return reachable[target]
            if target in new_reachable:
                return new_reachable[target]
                
        return None

    while time.time() < deadline:
        # Convert set to list for the finder
        elements_list = list(selected)
        
        # Check if any subset sums to 200
        conflict_subset = find_subset_summing_to_target(200, elements_list)
        
        if conflict_subset is None:
            break
            
        # Find the minimum element in the conflict subset to remove
        min_elem = min(conflict_subset)
        
        # Remove it
        selected.discard(min_elem)
        
        # Safety check: if selected becomes empty, stop (though task says nonempty, 
        # but if we remove all, we fail. Unlikely to remove all 1..100)
        if not selected:
             # Fallback: return a single large number that can't sum to 200 alone
             # Any number < 200 is fine on its own? No, subset of size 1 sums to x.
             # If x=200, it fails. But max is 100. So any single number is valid.
             selected.add(100)
             break

    # Ensure non-empty
    if not selected:
        return [100]
        
    return list(selected)

# EVOLVE-BLOCK-END
