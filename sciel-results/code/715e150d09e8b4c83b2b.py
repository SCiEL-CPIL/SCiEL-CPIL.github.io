    while time.time() - start_time < time_budget:
        if current_count == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
            
        # Pick a random cycle index
        # To improve efficiency, we can pick from a range of "hot" cycles?
        # No, just random.
        cycle_idx = random.randint(0, num_cycles - 1)
        
        # If not mono, try a few times or just skip?
        # If we skip, we might waste time.
        # Let's try to find a mono cycle.
        attempts = 0
        while not is_mono[cycle_idx] and attempts < 10:
            cycle_idx = random.randint(0, num_cycles - 1)
            attempts += 1
            
        if not is_mono[cycle_idx]:
            # No mono cycle found in 10 tries?
            # This implies current_count is very low or 0.
            # But we checked current_count > 0.
            # So we just continue.
            continue

        cycle_edges = cycles[cycle_idx]
        edge_idx = random.choice(cycle_edges)
        
        old_color = colors[edge_idx]
        new_color = (old_color + random.randint(1, 3)) % 4
        
        # Calculate delta
        delta = 0
        affected_cycles = edge_to_cycles[edge_idx]
        
        for c_idx in affected_cycles:
            was_mono = is_mono[c_idx]
            
            # Check new mono status
            # We know edge_idx will be new_color.
            # Check if all other edges in c_idx are new_color.
            is_new_mono = True
            for e in cycles[c_idx]:
                if e == edge_idx:
                    continue
                if colors[e] != new_color:
                    is_new_mono = False
                    break
            
            if was_mono and not is_new_mono:
                delta -= 1
            elif not was_mono and is_new_mono:
                delta += 1
            
        # Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            colors[edge_idx] = new_color
            current_count += delta
            
            # Update is_mono for affected cycles
            for c_idx in affected_cycles:
                c = colors[cycles[c_idx][0]]
                mono = True
                for e in cycles[c_idx][1:]:
                    if colors[e] != c:
                        mono = False
                        break
                is_mono[c_idx] = mono
                
        T *= cooling
