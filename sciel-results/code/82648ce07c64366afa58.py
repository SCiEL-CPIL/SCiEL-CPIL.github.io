<<<<<<< SEARCH
    # Strategy 4: Greedy with varying density patterns
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        selected = []
        selected_arr = np.array([], dtype=np.int64)
        selected_sums = set()
        selected_set = set()
        
        # Use a base step with occasional larger jumps
        base_step = rng.randint(2, 12)
        jump_prob = rng.uniform(0.05, 0.3)
        
        x = rng.randint(1, 50)
        while x <= 4096:
            if _check_collision_fast(selected_arr, selected_sums, x):
                pass
            else:
                _update_sums_fast(selected_arr, selected_sums, x)
                selected.append(x)
                selected_arr = np.append(selected_arr, x)
                selected_set.add(x)
            
            if rng.random() < jump_prob:
                x += base_step * rng.randint(2, 4)
            else:
                x += base_step
    
    if len(selected) > len(best):
        best = selected
=======
    # Strategy 4: Greedy with varying density patterns
    for attempt in range(1000):  # Increased attempts due to speedup
        if time.time() > deadline:
            break
        
        selected = []
        selected_arr = np.array([], dtype=np.int64)
        sums_used = bytearray(12289)
        selected_set = set()
        
        # Use a base step with occasional larger jumps
        base_step = rng.randint(2, 12)
        jump_prob = rng.uniform(0.05, 0.3)
        
        x = rng.randint(1, 50)
        while x <= 4096:
            if _check_collision_fast(selected_arr, sums_used, x):
                pass
            else:
                _update_sums_fast(selected_arr, sums_used, x)
                selected.append(x)
                selected_arr = np.append(selected_arr, x)
                selected_set.add(x)
            
            if rng.random() < jump_prob:
                x += base_step * rng.randint(2, 4)
            else:
                x += base_step
        
        if len(selected) > len(best):
            best = selected
>>>>>>> REPLACE

Update Strategy 5 in `construct()`:

