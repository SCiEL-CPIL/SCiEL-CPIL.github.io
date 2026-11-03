<<<<<<< SEARCH
    # Run SA from the best initial configuration
    sa_time_budget = DEADLINE * 0.70
    
    current_positions = spec_positions.copy()
    current_angles = spec_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.15
    T_min = 0.00005
    cooling = 0.99990
    step_size = 0.15
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0
    eval_count = 0
    no_improve_count = 0
    improve_threshold = 300
    last_restart = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 2.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.002: step_size = 0.002
            continue
            
        eval_count += 1
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            no_improve_count = 0
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            # Aggressive acceptance to explore
            if random.random() < math.exp(-delta / (T * 1.2)):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
            no_improve_count += 1
        
        # Adaptive step size
        if total_count % 50 == 0:
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.05:
                step_size *= 0.8
                if step_size < 0.002: step_size = 0.002
            elif acc_rate > 0.35:
                step_size *= 1.15
                if step_size > 0.35: step_size = 0.35
            accept_count = 0
            total_count = 0
        
        # Periodic restart from best if stuck
        if no_improve_count > improve_threshold:
            # Perturb best configuration
            current_positions = best_positions.copy()
            current_angles = best_angles.copy()
            # Add small random perturbations
            for i in range(N):
                current_positions[i, 0] += random.gauss(0, 0.05)
                current_positions[i, 1] += random.gauss(0, 0.05)
                current_angles[i] += random.gauss(0, 1.0)
            current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
            no_improve_count = 0
            T = max(T, 0.08)
            step_size = max(step_size, 0.12)
            last_restart = step_count
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.002: step_size = 0.002
=======
    # Run SA from the best initial configuration
    sa_time_budget = DEADLINE * 0.65
    
    current_positions = spec_positions.copy()
    current_angles = spec_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.12
    T_min = 0.00003
    cooling = 0.99995
    step_size = 0.12
    step_decay = 0.999995
    
    step_count = 0
    max_steps = 800000
    accept_count = 0
    total_count = 0
    eval_count = 0
    no_improve_count = 0
    improve_threshold = 500
    last_restart = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        # 20% chance of pair move, 80% single move
        if random.random() < 0.2:
            idx1 = random.randint(0, N - 1)
            idx2 = random.randint(0, N - 1)
            if idx2 == idx1:
                idx2 = (idx1 + 1) % N
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            for idx in [idx1, idx2]:
                dx = random.gauss(0, step_size)
                dy = random.gauss(0, step_size)
                da = random.gauss(0, 2.0)
                new_positions[idx, 0] += dx
                new_positions[idx, 1] += dy
                new_angles[idx] += da
            
            angles_rad_new = np.radians(new_angles)
            valid = True
            for idx in [idx1, idx2]:
                h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
                for j in range(N):
                    if j == idx:
                        continue
                    h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h_moved, h_other):
                        valid = False
                        break
                if not valid:
                    break
        else:
            idx = random.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 2.0)
            
            new_positions[idx, 0] += dx
            new_positions[idx, 1] += dy
            new_angles[idx] += da
            
            angles_rad_new = np.radians(new_angles)
            valid = True
            h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx:
                    continue
                h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h_moved, h_other):
                    valid = False
                    break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.002: step_size = 0.002
            continue
            
        eval_count += 1
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            no_improve_count = 0
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / (T * 1.0)):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
            no_improve_count += 1
        
        # Adaptive step size
        if total_count % 30 == 0:
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.05:
                step_size *= 0.85
                if step_size < 0.002: step_size = 0.002
            elif acc_rate > 0.30:
                step_size *= 1.12
                if step_size > 0.30: step_size = 0.30
            accept_count = 0
            total_count = 0
        
        # Periodic restart from best if stuck
        if no_improve_count > improve_threshold:
            current_positions = best_positions.copy()
            current_angles = best_angles.copy()
            for i in range(N):
                current_positions[i, 0] += random.gauss(0, 0.08)
                current_positions[i, 1] += random.gauss(0, 0.08)
                current_angles[i] += random.gauss(0, 1.5)
            current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
            no_improve_count = 0
            T = max(T, 0.06)
            step_size = max(step_size, 0.10)
            last_restart = step_count
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.002: step_size = 0.002
>>>>>>> REPLACE

