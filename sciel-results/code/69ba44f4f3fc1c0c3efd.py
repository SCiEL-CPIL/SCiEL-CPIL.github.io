            # Check if enough compatible nodes remain
            compat_count = 0
            for j in range(i + 1, n_subset): # Nodes after i in the current iteration order
                # Note: We are iterating i from pos to n_subset.
                # The "remaining" nodes for the next recursion are those with index > i in the `order` list?
                # No, the `order` list is a permutation.
                # In the loop `for i in range(pos, n_subset)`, we pick `order[i]`.
                # The candidates for the next step are `order[i+1]` to `order[n_subset-1]`.
                # We need to count how many of those are compatible with `current_mask | all_masks[global_idx]`.
                
                # This is expensive to do inside the loop.
                # Let's just use the simple count for now.
