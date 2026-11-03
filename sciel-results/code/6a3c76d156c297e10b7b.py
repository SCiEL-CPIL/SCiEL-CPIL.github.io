<<<<<<< SEARCH
    # --- Phase 2: L-BFGS-B Polish ---
    # We use a penalty method to handle the collision constraints.
    # Objective: Outer Side + Penalty * Sum(max(0, overlap_depth)^2)
    
    def objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        # Compute outer side
        side, cx, cy, phi = compute_outer_side_and_center(pos, ang)
        
        # Compute penalty for overlaps
        # We approximate overlap depth by checking distance between centers.
        # A more accurate penalty would use the SAT distance, but that's expensive.
        # For a polish, a simple center-distance penalty helps keep them apart.
        # However, center distance is not sufficient for rotated hexagons.
        # Let's just use the hard check in a loop? No, L-BFGS needs gradients.
        # We will use a "soft" collision check based on bounding boxes or similar?
        # Actually, if we start from a valid SA result, we can use a very small step size 
        # and just check validity, rejecting invalid steps in a custom callback? 
        # L-BFGS doesn't support that easily.
        
        # Alternative: Use Coordinate Descent Polish which is simpler and robust for valid configs.
        # The previous CD was slow. Let's make it faster.
        return side

    # Since L-BFGS with hard constraints is hard, we will do a faster Coordinate Descent.
    # We will iterate over each hexagon and each dimension, and find the best step.
    
    def polish_coordinate_descent(pos, ang, time_limit):
        current_side, _, _, _ = compute_outer_side_and_center(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        
        t_start = time.time()
        step = 0.1
        min_step = 0.001
        
        # We will try to move each hexagon slightly in each direction
        # and accept if it improves the side length and maintains validity.
        
        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_pos_cd.copy()
                        new_ang = best_ang_cd.copy()
                        
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * step
                        else:
                            new_ang[idx] += direction * step
                            
                        angles_rad_new = np.radians(new_ang)
                        valid = True
                        h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                        for j in range(N):
                            if j == idx: continue
                            h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                            if check_collision(h_moved, h_other):
                                valid = False
                                break
                            
                        if not valid:
                            continue
                            
                        new_side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                        
                        if new_side < best_side_cd - 1e-10:
                            best_pos_cd = new_pos
                            best_ang_cd = new_ang
                            best_side_cd = new_side
                            improved = True
                            # Don't break immediately, try other dims too?
                            # Breaking makes it faster to converge to a local min.
                            break
                        
                    if time.time() - t_start > time_limit:
                        return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    continue
                break
            
            if not improved:
                step *= 0.7
            else:
                # If improved, keep step size or slightly reduce to refine
                step *= 0.95
                if step < min_step:
                    step = min_step
            
        return best_side_cd, best_pos_cd, best_ang_cd

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time - 2.0)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
=======
    # --- Phase 2: Gradient Descent / L-BFGS Polish ---
    # Let's try a simple gradient descent with penalty for collisions.
    # Or just a faster coordinate descent.
    # The previous CD was O(N^3) per step.
    # Let's optimize CD.
    
    def polish_coordinate_descent(pos, ang, time_limit):
        current_side, _, _, _ = compute_outer_side_and_center(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        best_angles_rad_cd = np.radians(best_ang_cd)
        
        t_start = time.time()
        step = 0.1
        min_step = 0.0001
        
        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            # Shuffle order to avoid bias?
            indices = list(range(N))
            random.shuffle(indices)
            for idx in indices:
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_pos_cd.copy()
                        new_ang = best_ang_cd.copy()
                        
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * step
                        else:
                            new_ang[idx] += direction * step
                            
                        # Check validity
                        angles_rad_new = best_angles_rad_cd.copy()
                        if dim == 2:
                            angles_rad_new[idx] = np.radians(new_ang[idx])
                        
                        pos_moved = new_pos[idx]
                        ang_moved = angles_rad_new[idx]
                        valid = True
                        for j in range(N):
                            if j == idx: continue
                            if fast_check_collision_vec(pos_moved, ang_moved, new_pos[j], angles_rad_new[j]):
                                valid = False
                                break
                            
                        if not valid:
                            continue
                            
                        new_side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                        
                        if new_side < best_side_cd - 1e-10:
                            best_pos_cd = new_pos
                            best_ang_cd = new_ang
                            best_angles_rad_cd = angles_rad_new
                            best_side_cd = new_side
                            improved = True
                            break
                        
                    if time.time() - t_start > time_limit:
                        return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    continue
                break
            
            if not improved:
                step *= 0.7
            else:
                step *= 0.95
                if step < min_step:
                    step = min_step
            
        return best_side_cd, best_pos_cd, best_ang_cd

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time - 2.0)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
>>>>>>> REPLACE

