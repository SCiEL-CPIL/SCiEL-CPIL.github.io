<<<<<<< SEARCH
    # Strategy: Run SA for 65% of the time, then Polish for the rest
    sa_time_budget = DEADLINE * 0.65
    
    # Single robust SA run with adaptive parameters
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.3
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 200000
    accept_count = 0
    total_count = 0
    
    # Helper for multi-particle moves
    def generate_move(current_pos, current_ang, step_sz):
        # 70% single, 20% pair, 10% triple
        r = random.random()
        if r < 0.7:
            idxs = [random.randint(0, N - 1)]
        elif r < 0.9:
            idxs = random.sample(range(N), 2)
        else:
            idxs = random.sample(range(N), 3)
            
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        
        for idx in idxs:
            dx = random.gauss(0, step_sz)
            dy = random.gauss(0, step_sz)
            da = random.gauss(0, 5.0)
            new_pos[idx, 0] += dx
            new_pos[idx, 1] += dy
            new_ang[idx] += da
            
        return new_pos, new_ang, idxs

    def check_validity(new_pos, new_ang, moved_idxs):
        angles_rad_new = np.radians(new_ang)
        # Check moved against all others
        for idx in moved_idxs:
            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx:
                    continue
                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h_moved, h_other):
                    return False
        # Check moved against moved (if more than 1)
        if len(moved_idxs) > 1:
            for i_idx in range(len(moved_idxs)):
                for j_idx in range(i_idx + 1, len(moved_idxs)):
                    i = moved_idxs[i_idx]
                    j = moved_idxs[j_idx]
                    h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad_new[i])
                    h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        return False
        return True

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        new_positions, new_angles, moved_idxs = generate_move(current_positions, current_angles, step_size)
        
        valid = check_validity(new_positions, new_angles, moved_idxs)
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
            continue
            
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
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
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        
        # Adaptive step size
        if total_count % 500 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.1:
                step_size *= 0.8
                if step_size < 0.005: step_size = 0.005
            elif acc_rate > 0.5:
                step_size *= 1.2
                if step_size > 1.0: step_size = 1.0
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005
=======
    # Strategy: Run multiple SA restarts for 70% of the time, then Polish for the rest
    sa_time_budget = DEADLINE * 0.70
    
    def run_sa(initial_pos, initial_ang, time_limit, T_init, step_init):
        """Run SA for a given time limit, return best found."""
        current_positions = initial_pos.copy()
        current_angles = initial_ang.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        
        best_side_local = current_side
        best_positions_local = current_positions.copy()
        best_angles_local = current_angles.copy()
        best_phi_local = current_phi
        best_cx_local = current_cx
        best_cy_local = current_cy
        
        T = T_init
        T_min = 0.0001
        cooling = 0.9999
        step_size = step_init
        step_decay = 0.99998
        
        step_count = 0
        accept_count = 0
        total_count = 0
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            step_count += 1
            total_count += 1
            
            # Generate move: 60% single, 25% pair, 15% triple
            r = random.random()
            if r < 0.6:
                idxs = [random.randint(0, N - 1)]
            elif r < 0.85:
                idxs = random.sample(range(N), 2)
            else:
                idxs = random.sample(range(N), 3)
                
            new_pos = current_positions.copy()
            new_ang = current_angles.copy()
            
            for idx in idxs:
                dx = random.gauss(0, step_size)
                dy = random.gauss(0, step_size)
                da = random.gauss(0, 8.0)
                new_pos[idx, 0] += dx
                new_pos[idx, 1] += dy
                new_ang[idx] += da
            
            # Check validity
            angles_rad_new = np.radians(new_ang)
            valid = True
            for idx in idxs:
                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                for j in range(N):
                    if j == idx:
                        continue
                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h_moved, h_other):
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(idxs) > 1:
                for i_idx in range(len(idxs)):
                    for j_idx in range(i_idx + 1, len(idxs)):
                        i = idxs[i_idx]
                        j = idxs[j_idx]
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad_new[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
            
            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.005: step_size = 0.005
                continue
            
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
            
            if new_side < current_side:
                current_positions = new_pos
                current_angles = new_ang
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side_local:
                    best_side_local = new_side
                    best_positions_local = new_pos.copy()
                    best_angles_local = new_ang.copy()
                    best_phi_local = new_phi
                    best_cx_local = new_cx
                    best_cy_local = new_cy
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    current_positions = new_pos
                    current_angles = new_ang
                    current_side = new_side
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                    accept_count += 1
                    if new_side < best_side_local:
                        best_side_local = new_side
                        best_positions_local = new_pos.copy()
                        best_angles_local = new_ang.copy()
                        best_phi_local = new_phi
                        best_cx_local = new_cx
                        best_cy_local = new_cy
            
            # Adaptive step size every 300 steps
            if total_count % 300 == 0:
                acc_rate = accept_count / total_count
                if acc_rate < 0.05:
                    step_size *= 0.7
                    if step_size < 0.005: step_size = 0.005
                elif acc_rate > 0.4:
                    step_size *= 1.3
                    if step_size > 1.5: step_size = 1.5
                accept_count = 0
                total_count = 0
                
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
        
        return best_side_local, best_positions_local, best_angles_local, best_phi_local, best_cx_local, best_cy_local

    # Multiple SA restarts with different initial conditions
    num_restarts = 5
    restart_time = sa_time_budget / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - t0 >= sa_time_budget:
            break
        
        # Perturb the best solution for restarts after the first
        if restart == 0:
            init_pos = base_positions.copy()
            init_ang = base_angles.copy()
            T_init = 0.8
            step_init = 0.5
        else:
            # Perturb from current best
            init_pos = best_positions.copy()
            init_ang = best_angles.copy()
            for i in range(N):
                init_pos[i, 0] += random.gauss(0, 0.3)
                init_pos[i, 1] += random.gauss(0, 0.3)
                init_ang[i] += random.gauss(0, 10.0)
            T_init = 0.4
            step_init = 0.3
        
        rs, rp, ra, rphi, rcx, rcy = run_sa(init_pos, init_ang, restart_time, T_init, step_init)
        
        if rs < best_side:
            best_side = rs
            best_positions = rp
            best_angles = ra
            best_phi = rphi
            best_cx = rcx
            best_cy = rcy
>>>>>>> REPLACE

Now let me also improve the polish phase:

