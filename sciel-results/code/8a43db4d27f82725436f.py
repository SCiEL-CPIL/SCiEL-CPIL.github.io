        i = 0
        while i < len(order):
            idx = order[i]
            # Check if idx is still valid
            if idx >= len(selected_list):
                i += 1
                continue
            c = selected_list[idx]
            if np.min(cover_count[balls[c]]) > 1:
                # Remove
                cover_count[balls[c]] -= 1
                selected_list.pop(idx)
                selected_set.discard(c)
                improved = True
                # All indices in order > idx are now shifted down by 1
                # We don't need to update order explicitly if we just skip the next one?
                # No, indices > idx in order now point to wrong elements.
                # Simplest: restart the pass.
                break 
            i += 1
