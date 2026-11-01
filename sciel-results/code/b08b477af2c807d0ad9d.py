        if new_val > max_val:
            new_scope = new_val
        elif old_val == max_val and new_val < old_val:
            # Recalculate
            ...
        else:
            new_scope = max_val
