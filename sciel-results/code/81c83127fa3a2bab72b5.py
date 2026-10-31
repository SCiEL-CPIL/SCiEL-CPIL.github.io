<<<<<<< SEARCH
def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity, conservatively."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        # Boundary: clamp radii to fit in square
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.9999
                valid = False
        # Overlap: only shrink the smaller circle, minimally
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if radii[i] <= radii[j]:
                        # Shrink i by at most half the overlap
                        shrink = min(overlap * 0.51, radii[i] * 0.1)
                        radii[i] -= shrink
                    else:
                        shrink = min(overlap * 0.51, radii[j] * 0.1)
                        radii[j] -= shrink
                    if radii[i] < 1e-6:
                        radii[i] = 1e-6
                    if radii[j] < 1e-6:
                        radii[j] = 1e-6
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r * 0.9999))

    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """SLSQP polishing with explicit constraints."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii, deadline)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        return -np.sum(r)

    def make_constraint(i, j):
        def constraint(x):
            p = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - r[i] - r[j]
        return constraint

    # Boundary constraints
    boundary_cons = []
    for i in range(N):
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i+1]})

    # Overlap constraints
    overlap_cons = []
    for i in range(N):
        for j in range(i + 1, N):
            overlap_cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=boundary_cons + overlap_cons,
                          options={'maxiter': 400, 'ftol': 1e-14})
        if time.time() < deadline:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt, deadline)
            if _is_valid(p_opt, r_opt):
                s = np.sum(r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score


def _gradient_polish(pos, radii, deadline):
    """Gradient-based polishing with smooth penalty."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii, deadline)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    def objective(x):
        return _objective(x, deadline)

    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))

    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
        if time.time() < deadline:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt, deadline)
            if _is_valid(p_opt, r_opt):
                s = np.sum(r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score
=======
def _project_valid(pos, radii, max_iter=1000):
    """
    Project a potentially invalid configuration to a valid one.
    Strategy: Iteratively resolve overlaps by moving centers apart or shrinking radii.
    This guarantees a valid output if it converges.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Initial boundary clamp
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = min(radii[i], max_r * 0.9999)
        radii[i] = max(radii[i], 1e-6)
        # Clamp position to be inside [r, 1-r]
        pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
        pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])

    for _ in range(max_iter):
        changed = False
        # Resolve overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # If dist is 0, pick a random direction
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1e-9
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Move centers apart by half the overlap
                    push = overlap * 0.51
                    pos[i, 0] += dx * push
                    pos[i, 1] += dy * push
                    pos[j, 0] -= dx * push
                    pos[j, 1] -= dy * push
                    
                    # Re-clamp to boundaries
                    for k in [i, j]:
                        max_r = min(pos[k, 0], 1.0 - pos[k, 0], pos[k, 1], 1.0 - pos[k, 1])
                        if radii[k] > max_r:
                            radii[k] = max_r * 0.9999
                            radii[k] = max(radii[k], 1e-6)
                        pos[k, 0] = np.clip(pos[k, 0], radii[k], 1.0 - radii[k])
                        pos[k, 1] = np.clip(pos[k, 1], radii[k], 1.0 - radii[k])
                    changed = True
        
        if not changed:
            break
    
    return pos, radii


def _gradient_polish(pos, radii, deadline):
    """
    Gradient-based polishing with projection.
    We maximize sum of radii by taking gradient steps on radii and positions,
    then projecting back to feasibility.
    """
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _project_valid(pos, radii, max_iter=100)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Simple gradient ascent on radii (since we want to maximize sum of radii)
    # We also adjust positions to allow for larger radii.
    # For simplicity and robustness, we primarily expand radii and let projection fix positions.
    
    step_size = 0.01
    for _ in range(500):
        if time.time() > deadline:
            break
            
        # Try to increase all radii slightly
        radii_new = radii + step_size
        
        # Project to valid
        pos_new, radii_new = _project_valid(pos, radii_new, max_iter=100)
        
        # Check if this improved the score
        current_score = np.sum(radii_new)
        if current_score > best_score:
            best_score = current_score
            best_pos = pos_new.copy()
            best_radii = radii_new.copy()
            pos = pos_new.copy()
            radii = radii_new.copy()
            
            # If we made progress, maybe increase step size slightly
            step_size *= 1.05
        else:
            # If no progress, decrease step size
            step_size *= 0.9
            
            # Also try local position adjustments to unlock larger radii
            # This is a simple heuristic: nudge positions randomly and project
            pos_trial = pos + np.random.normal(0, 0.01, size=(N, 2))
            pos_trial, radii_trial = _project_valid(pos_trial, radii, max_iter=100)
            trial_score = np.sum(radii_trial)
            if trial_score > best_score:
                best_score = trial_score
                best_pos = pos_trial.copy()
                best_radii = radii_trial.copy()
                pos = pos_trial.copy()
                radii = radii_trial.copy()
                
    return best_pos, best_radii, best_score
>>>>>>> REPLACE

