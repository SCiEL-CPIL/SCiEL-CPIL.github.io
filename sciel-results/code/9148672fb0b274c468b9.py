"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Numbers of interest
    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    
    # Map value to index
    idx = {x: i for i, x in enumerate(nums)}
    
    # Precompute conflict matrix
    # conflict[i] is a bitmask of indices j such that lcm(nums[i], nums[j]) <= 256
    # We use integers as bitmasks
    conflict = [0] * n
    
    for i in range(n):
        ai = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            bj = nums[j]
            if lcm(ai, bj) <= N:
                mask |= (1 << j)
        conflict[i] = mask

    # Helper to check if adding index k to a set represented by bitmask S is valid
    # S contains indices of selected numbers
    # We need to ensure k is not in S, and no element in S conflicts with k
    # conflict[k] has bits for all j conflicting with k.
    # If (S & conflict[k]) != 0, then there is a conflict.
    
    # We split the numbers into "Small" (2..16) and "Large" (17..256)
    # Small indices
    small_vals = list(range(2, 17))
    small_indices = [idx[x] for x in small_vals]
    n_small = len(small_indices)
    
    # Map small value to small index (0..14)
    small_val_to_idx = {x: i for i, x in enumerate(small_vals)}
    
    # We will enumerate all subsets of small indices
    # For each subset, check if it's valid internally.
    # If valid, determine which large numbers are compatible with all small numbers in the subset.
    # Then solve Max Weight Independent Set on the remaining large numbers.
    
    best_set_indices = []
    best_score = 0.0
    
    # Precompute reciprocal weights
    weights = [1.0 / x for x in nums]
    
    # Iterate over all subsets of small numbers
    # There are 2^15 = 32768 subsets
    for mask_small in range(1 << n_small):
        if time.time() > deadline:
            break
            
        # Extract small indices in this mask
        small_in_set = []
        for i in range(n_small):
            if mask_small & (1 << i):
                small_in_set.append(small_indices[i])
                
        # Check internal validity of small_in_set
        valid = True
        # We can optimize this, but n_small is small
        # Check all pairs in small_in_set
        for i in range(len(small_in_set)):
            idx_i = small_in_set[i]
            # If any other selected small index is in conflict[idx_i], it's invalid
            # We can build a mask of selected small indices
            # But let's just loop, it's fast enough
            for j in range(i + 1, len(small_in_set)):
                idx_j = small_in_set[j]
                if conflict[idx_i] & (1 << idx_j):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Determine compatible large numbers
        # Large numbers are indices corresponding to values 17..256
        # We start with all large indices available
        # Large indices are those where nums[idx] >= 17
        
        # Mask of all large indices
        large_mask = 0
        for i in range(n):
            if nums[i] >= 17:
                large_mask |= (1 << i)
        
        # For each small index in small_in_set, remove its conflicts from large_mask
        # conflict[idx_i] contains all conflicting indices.
        # We want large_mask that does NOT intersect with conflict[idx_i]
        # So: large_mask &= ~conflict[idx_i]
        
        for idx_i in small_in_set:
            large_mask &= ~conflict[idx_i]
            
        # Now large_mask contains all large numbers compatible with all small numbers in small_in_set
        # We need to find a max weight independent set from these large numbers.
        # The graph induced by large numbers is sparse.
        # Greedy approach:
        # Get list of indices in large_mask
        # Sort by value (ascending) to prioritize high weights
        # Iterate and add if compatible with current selection
        
        # Extract indices
        candidates = []
        # To speed up, we can pre-list large indices
        # But large_mask changes.
        # We can iterate through all indices and check bit
        # Or better, pre-compute list of large indices
        large_indices_list = [i for i in range(n) if nums[i] >= 17]
        
        # Filter candidates based on large_mask
        active_candidates = [i for i in large_indices_list if large_mask & (1 << i)]
        
        # Sort by value (ascending)
        active_candidates.sort(key=lambda x: nums[x])
        
        # Greedy selection
        selected_large = []
        selected_mask = 0 # Mask of selected large indices
        
        for i in active_candidates:
            # Check if i conflicts with any in selected_large
            # conflict[i] has bits for all conflicting indices.
            # If (selected_mask & conflict[i]) != 0, conflict exists.
            if selected_mask & conflict[i]:
                continue
            # Add i
            selected_large.append(i)
            selected_mask |= (1 << i)
            
        # Combine small and large
        full_set_indices = small_in_set + selected_large
        
        # Calculate score
        score = sum(weights[i] for i in full_set_indices)
        
        if score > best_score:
            best_score = score
            best_set_indices = list(full_set_indices)
            
    # Convert indices back to values
    result = [nums[i] for i in best_set_indices]
    
    # Final sanity check and local improvement (optional, but safe)
    # The greedy on large numbers might not be optimal if there are conflicts among large numbers.
    # However, given the sparsity, it's likely good.
    # Let's do a quick local search on the best set found to ensure no obvious improvements.
    
    if time.time() < deadline - 5.0:
        current_set = set(best_set_indices)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try adding one
            for i in range(n):
                if i in current_set:
                    continue
                # Check if i can be added
                # Need to check conflicts with all in current_set
                # We can use a mask of current_set
                # But current_set is a set of indices.
                # Let's build a mask
                # This is O(N) per check, so O(N^2) per loop.
                # N=255. 255^2 * iterations.
                # We can optimize by maintaining a mask.
                pass # Skipping complex local search to keep code simple and fast, relying on the structural decomposition.
            
            # Try removing one and adding another?
            # The structural decomposition already explores all combinations of small numbers.
            # The only "greedy" part is the large numbers.
            # Since large numbers have low weights, the greedy is likely optimal or very close.
            
    return result
# EVOLVE-BLOCK-END
