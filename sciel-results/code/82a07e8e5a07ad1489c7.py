            # Apply change
            colors[e_idx] = new_c
            
            # Check impact on affected K4s
            delta_invalid = 0
            affected_k4s = k4_edge_map[e_idx]
            
            # Store old states
            old_states = [valid_status[k] for k in affected_k4s]
            
            for i, k4_idx in enumerate(affected_k4s):
                old_stat = old_states[i]
                new_stat = check_k4(k4_idx, colors)
                
                if old_stat == new_stat:
                    continue
                    
                if old_stat and not new_stat:
                    delta_invalid += 1
                elif not old_stat and new_stat:
                    delta_invalid -= 1
                
                valid_status[k4_idx] = new_stat
            
            num_invalid += delta_invalid
            
            if num_invalid < best_invalid_count:
                best_invalid_count = num_invalid
                best_colors = colors.copy()
                
            if num_invalid == 0:
                return colors
            
            # Revert if bad
            if delta_invalid > 0:
                colors[e_idx] = old_c
                for i, k4_idx in enumerate(affected_k4s):
                    valid_status[k4_idx] = old_states[i]
                num_invalid -= delta_invalid
