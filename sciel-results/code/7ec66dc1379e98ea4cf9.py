<<<<<<< SEARCH
def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 1: Low penalty, focus on layout
    penalty = 5.0
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 50.0
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: High penalty, Delaunay-based for speed
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 200.0
        circles = params.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, contact_edges),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_delaunay(params, penalty, 1e-5, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Final polish with very high penalty
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 1000.0
        circles = best_params.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6, contact_edges),
                          options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < _objective_delaunay(best_params, penalty, 1e-6, contact_edges):
                best_params = res.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass

    # Validate and fix
    best_circles = best_params.reshape(N, 3)
    best_circles = _fix_boundaries(best_circles)
    if not _validate(best_circles):
        # Try to fix by reducing radii slightly
        best_circles[:, 2] *= 0.99
        best_circles = _fix_boundaries(best_circles)
        if not _validate(best_circles):
            # Fall back to original
            best_circles = xyz0.copy()

    return best_circles
=======
def _pbd_relax(circles, deadline, max_iters=2000, grow_rate=0.001):
    """
    Physics-based Position Based Dynamics relaxation.
    - Treat circles as particles with mass ~ r^2
    - Apply outward pressure to grow radii
    - Resolve overlaps and boundary constraints via Gauss-Seidel
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    # Mass proportional to radius squared (minimum mass to avoid zero)
    mass = np.maximum(r, 0.01) ** 2
    mass_inv = 1.0 / mass
    
    # Contact edges for efficiency
    contact_edges = _get_contact_edges(circles)
    if contact_edges is None:
        contact_edges = []
    
    best_r_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Grow radii slightly
        r = np.minimum(r + grow_rate, 0.45)
        
        # Multiple Gauss-Seidel passes for constraint resolution
        for _pass in range(3):
            # Resolve pairwise overlaps
            if contact_edges:
                for (i, j) in contact_edges:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        if dist < 1e-12:
                            dx, dy = 1e-8, 0.0
                            dist = 1e-8
                        # Push apart proportional to inverse mass
                        overlap = min_dist - dist
                        total_inv = mass_inv[i] + mass_inv[j]
                        push_i = overlap * mass_inv[i] / total_inv
                        push_j = overlap * mass_inv[j] / total_inv
                        nx, ny = dx/dist, dy/dist
                        x[i] += push_i * nx
                        y[i] += push_i * ny
                        x[j] -= push_j * nx
                        y[j] -= push_j * ny
            else:
                # Fallback: check all pairs
                for i in range(n):
                    for j in range(i+1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            if dist < 1e-12:
                                dx, dy = 1e-8, 0.0
                                dist = 1e-8
                            overlap = min_dist - dist
                            total_inv = mass_inv[i] + mass_inv[j]
                            push_i = overlap * mass_inv[i] / total_inv
                            push_j = overlap * mass_inv[j] / total_inv
                            nx, ny = dx/dist, dy/dist
                            x[i] += push_i * nx
                            y[i] += push_i * ny
                            x[j] -= push_j * nx
                            y[j] -= push_j * ny
            
            # Resolve boundary constraints
            for i in range(n):
                ri = r[i]
                if x[i] < ri:
                    x[i] = ri
                elif x[i] > 1 - ri:
                    x[i] = 1 - ri
                if y[i] < ri:
                    y[i] = ri
                elif y[i] > 1 - ri:
                    y[i] = 1 - ri
        
        # Update contact edges periodically
        if it % 50 == 0:
            circles_temp = np.column_stack([x, y, r])
            new_edges = _get_contact_edges(circles_temp)
            if new_edges is not None:
                contact_edges = new_edges
        
        # Track best
        current_sum = np.sum(r)
        if current_sum > best_r_sum:
            best_r_sum = current_sum
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    return np.column_stack([best_x, best_y, best_r])


def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty + PBD."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 0: PBD relaxation to find a good feasible layout
    remaining = deadline - time.time()
    if remaining > 20:
        try:
            pbd_circles = _pbd_relax(xyz0, deadline, max_iters=1500, grow_rate=0.0008)
            pbd_val = np.sum(pbd_circles[:, 2])
            if pbd_val > best_score:
                best_score = pbd_val
                best_params = pbd_circles.flatten().copy()
                params = best_params.copy()
        except Exception:
            pass

    # Phase 1: Low penalty, focus on layout
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(5.0, 1e-3),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, 5.0, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(50.0, 1e-4),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, 50.0, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2.5: PBD refinement after L-BFGS
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            current_circles = params.reshape(N, 3)
            pbd_circles = _pbd_relax(current_circles, deadline, max_iters=800, grow_rate=0.0005)
            pbd_val = np.sum(pbd_circles[:, 2])
            if pbd_val > best_score:
                best_score = pbd_val
                best_params = pbd_circles.flatten().copy()
                params = best_params.copy()
        except Exception:
            pass

    # Phase 3: High penalty, Delaunay-based for speed
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 200.0
        circles = params.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, contact_edges),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_delaunay(params, penalty, 1e-5, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Final polish with very high penalty
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 1000.0
        circles = best_params.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6, contact_edges),
                          options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < _objective_delaunay(best_params, penalty, 1e-6, contact_edges):
                best_params = res.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass

    # Final PBD micro-refinement
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            current_circles = best_params.reshape(N, 3)
            pbd_circles = _pbd_relax(current_circles, deadline, max_iters=500, grow_rate=0.0003)
            pbd_val = np.sum(pbd_circles[:, 2])
            if pbd_val > best_score:
                best_score = pbd_val
                best_params = pbd_circles.flatten().copy()
        except Exception:
            pass

    # Validate and fix
    best_circles = best_params.reshape(N, 3)
    best_circles = _fix_boundaries(best_circles)
    if not _validate(best_circles):
        # Try to fix by reducing radii slightly
        best_circles[:, 2] *= 0.99
        best_circles = _fix_boundaries(best_circles)
        if not _validate(best_circles):
            # Fall back to original
            best_circles = xyz0.copy()

    return best_circles
>>>>>>> REPLACE

