    # ... inside while loop ...
    if not improved:
        # Perform a neutral move to explore
        # Pick a random row and position
        r_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k)
        current_val = current_rows[r_idx][pos_idx]
        
        low = current_rows[r_idx][pos_idx - 1] + 1
        high = current_rows[r_idx][pos_idx + 1] - 1 if pos_idx < k else current_val + 5
        
        if low <= high:
            new_val = random.randint(low, high)
            if new_val != current_val:
                if try_move(r_idx, pos_idx, new_val):
                    # Check if max changed (it shouldn't, unless we accidentally increased it? 
                    # No, we are in a valid state. If we change a non-max element, max stays same or decreases?
                    # It can only decrease if we were changing a max element.
                    # Since we picked random, it might be a max element.
                    new_max = max(max(r) for r in current_rows)
                    if new_max < current_max:
                        current_max = new_max
                        if current_max < best_max:
                            best_max = current_max
                            best_rows = [r[:] for r in current_rows]
                    # We keep the move regardless (neutral or improving) to explore
                    # But if it increased max? 
                    # The move is valid, so no conflicts.
                    # Can max increase? Yes, if we increased a value.
                    # We should probably revert if max increases.
                    if new_max > current_max:
                        try_move(r_idx, pos_idx, current_val)
                    else:
                        current_max = new_max
