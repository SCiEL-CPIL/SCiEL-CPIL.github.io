def solve_n_single(n, deadline):
    # ... initialization ...
    
    # Local search loop
    # Stop if time > deadline or success
    # But we want to allow restarts.
    # So we should stop after a certain number of iterations or time slice.
    # Let's say 2 seconds per restart.
    
    sub_deadline = time.time() + 2.0
    if sub_deadline > deadline:
        sub_deadline = deadline
        
    # ... local search loop with sub_deadline ...
    
    if total_c4 == 0:
        return format_result(n, colors)
    return None
