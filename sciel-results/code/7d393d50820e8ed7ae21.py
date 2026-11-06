    best_overall_mask = 0
    best_overall_fitness = -1
    
    for restart in range(5):
        # Reset
        current_mask = full_mask
        curr_viol = count_violations(current_mask)
        curr_size = N
        curr_fitness = curr_size - PENALTY * curr_viol
        
        T = 10.0
        cooling = 0.999
        steps = 20000 # Reduce steps per restart to allow more restarts
        
        for step in range(steps):
             # ... same logic ...
             
        if curr_fitness > best_overall_fitness:
             best_overall_fitness = curr_fitness
             best_overall_mask = current_mask
             
    # Extract from best_overall_mask
