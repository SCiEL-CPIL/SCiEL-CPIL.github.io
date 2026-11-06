    current_best = base_colors
    current_count = base_count
    
    for target_k in range(base_count - 1, 1, -1):
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break
            
        time_per_target = min(remaining_time, 5.0)
        
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
            # Use current_best as the starting point
            sa_colors, energy = sa_optimize(current_best, target_k, time_per_target / 3.0)
            
            if energy == 0:
                current_best = sa_colors
                current_count = target_k
                found = True
                break
        
        if found:
            continue
        else:
            # If we fail to reduce to target_k, it might be impossible or just hard.
            # We can break to save time for other restarts?
            # Or just continue.
            pass

    return current_best
