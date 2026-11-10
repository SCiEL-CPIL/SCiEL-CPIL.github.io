    while time.time() - t_start < WALL_LIMIT:
        # Choose a parameter to perturb
        if random.random() < 0.1:
            param_type = 'dc'
            old_val = dc
            dc += random.uniform(-step_size, step_size)
        else:
            param_type = 'amp'
            mode = random.choice(modes)
            old_val = amps[mode]
            amps[mode] += random.uniform(-step_size, step_size)
            
        cell = get_cell()
        score = _gap_ratio(cell, n_pw, kpts)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            no_improve_count = 0
        else:
            # Revert
            if param_type == 'dc':
                dc = old_val
            else:
                amps[mode] = old_val
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                step_size *= 0.9
                no_improve_count = 0
                if step_size < 0.01:
                    break
