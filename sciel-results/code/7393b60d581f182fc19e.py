def search_coloring(n, start_time, timeout, margin):
    # ... precomputation ...
    
    for r in range(max_restarts):
        # ... init ...
        
        # Calculate initial violations
        red_violations = np.sum(red_count == 4)
        blue_violations = np.sum(blue_count == 10)
        total_violations = red_violations + blue_violations
        
        # Local search
        while time.time() < start_time + timeout - margin:
            if total_violations == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_violations = 0
            
            # Check 4-sets
            for f_idx in triple_to_fours[t_idx]:
                old_cnt = red_count[f_idx]
                if old_color == 0: # Red -> Blue
                    new_cnt = old_cnt - 1
                    if old_cnt == 4: delta_violations -= 1
                    elif new_cnt == 4: delta_violations += 1 # Impossible, 3->2 or 4->3
                else: # Blue -> Red
                    new_cnt = old_cnt + 1
                    if old_cnt == 4: delta_violations -= 1
                    elif new_cnt == 4: delta_violations += 1
                    
                # Apply
                red_count[f_idx] = new_cnt
                
            # Check 5-sets
            for f_idx in triple_to_fives[t_idx]:
                old_cnt = blue_count[f_idx]
                if old_color == 1: # Blue -> Red
                    new_cnt = old_cnt - 1
                    if old_cnt == 10: delta_violations -= 1
                    elif new_cnt == 10: delta_violations += 1
                else: # Red -> Blue
                    new_cnt = old_cnt + 1
                    if old_cnt == 10: delta_violations -= 1
                    elif new_cnt == 10: delta_violations += 1
                    
                blue_count[f_idx] = new_cnt
                
            new_total = total_violations + delta_violations
            
            if new_total < total_violations:
                total_violations = new_total
                colors[t_idx] = new_color
                # Keep change
            else:
                # Revert
                colors[t_idx] = old_color
                for f_idx in triple_to_fours[t_idx]:
                    if old_color == 0:
                        red_count[f_idx] += 1
                    else:
                        red_count[f_idx] -= 1
                for f_idx in triple_to_fives[t_idx]:
                    if old_color == 1:
                        blue_count[f_idx] += 1
                    else:
                        blue_count[f_idx] -= 1
                        
    return None
