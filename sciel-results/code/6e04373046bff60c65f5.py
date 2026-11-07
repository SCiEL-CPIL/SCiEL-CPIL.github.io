        # Initialize S and S_set
        S = random.sample(range(1, max_val + 1), current_target_size)
        S_set = set(S)
        
        # In loop:
        # Remove old_val from S_set
        S_set.remove(old_val)
        S_set.add(new_val)
        # Check new_val in S_set BEFORE adding?
        # No, we check `if new_val in S` which uses list.
        # If we use S_set:
        # if new_val in S_set: continue (but we removed old_val, so if new_val == old_val, it's not in S_set? No, we removed it. So if new_val == old_val, it's not in S_set. But we skip new_val == old_val anyway.)
        # So:
        # if new_val in S_set: continue
        # Then S_set.add(new_val)
