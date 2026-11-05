    best_n = 9
    best_colors = quadratic_residue_coloring(9)
    
    # Try large n first? Or incremental?
    # Incremental is safer. But we want to maximize n.
    # If we have time, try 34, 33, ... down to 10.
    # If 34 fails, try 33, etc.
    
    # We have ~140s.
    # Let's allocate time based on n.
    # n=34: 60s
    # n=33: 30s
    # n=32: 20s
    # ...
    
    time_budget = {
        34: 60,
        33: 30,
        32: 20,
        31: 15,
        30: 10,
        29: 5,
        28: 5,
        27: 5,
        26: 5,
        25: 5,
        24: 5,
    }
    
    # Start from high n
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
            
        budget = time_budget.get(n, 5)
        # Don't exceed remaining time
        actual_budget = min(budget, deadline - time.time() - 2)
        
        if actual_budget < 1:
            break
            
        try:
            structures = build_structure(n)
            # Try a few restarts
            found = False
            attempts = 3
            for i in range(attempts):
                if time.time() > deadline - 2:
                    break
                t_start = time.time()
                t_limit = t_start + (actual_budget / attempts)
                
                # Different inits
                if i == 0:
                    init = quadratic_residue_coloring(n)
                else:
                    init = np.random.randint(0, 2, len(structures[1]), dtype=np.int8)
                    
                result = local_search(n, t_limit, init_colors=init, structures=structures)
                
                if result is not None:
                    # Verify
                    colors_np = np.array(result, dtype=np.int8)
                    if count_violations(colors_np, structures[3], structures[4]) == 0:
                        best_n = n
                        best_colors = result
                        found = True
                        break # Found valid for this n, go to smaller n? No, we want MAX n.
                        # So if we found valid for n, we can stop trying smaller n?
                        # Yes, because we are iterating downwards.
                        # But wait, if we found valid for 34, we return 34.
                        # If we failed 34, we try 33.
                        
            if found:
                break # We found the largest n (since we go down)
                
        except Exception:
            pass
            
    return {'n': best_n, 'colors': best_colors}
