<<<<<<< SEARCH
def construct():
    t0 = time.time()
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _max_overlap(h0)
    
    # Try SLSQP to minimize max overlap
    try:
        # Use the smooth objective for gradient-based optimization
        res = minimize(
            _objective_smooth,
            h0,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                          'jac': lambda x: np.ones(N)}],
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        elapsed = time.time() - t0
        if elapsed < DEADLINE:
            h_candidate = res.x
            # Project to valid range and fix sum
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            # Adjust to ensure sum is exactly N/2
            diff = N / 2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-10:
                # Distribute adjustment
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                # Re-fix sum
                diff2 = N / 2.0 - np.sum(h_candidate)
                h_candidate += diff2 / N
            
            val = _max_overlap(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
    except Exception:
        pass
    
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 5:
        # Try SLSQP with direct objective
        try:
            res2 = minimize(
                _objective,
                h0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            h_candidate = res2.x
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-10:
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                diff2 = N / 2.0 - np.sum(h_candidate)
                h_candidate += diff2 / N
            
            val = _max_overlap(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
    
    # Ensure exact sum
    final_h = best_h.copy()
    final_h = np.clip(final_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(final_h)
    if abs(diff) > 1e-12:
        # Add to elements with room
        for i in range(N):
            if diff > 0 and final_h[i] < 1.0:
                add = min(diff, 1.0 - final_h[i])
                final_h[i] += add
                diff -= add
            elif diff < 0 and final_h[i] > 0.0:
                sub = min(-diff, final_h[i])
                final_h[i] -= sub
                diff += sub
            if abs(diff) < 1e-12:
                break
    
    return final_h
=======
def _project_to_valid(h):
    """Project h to [0,1]^N with sum exactly N/2."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Iterate to fix sum while respecting bounds
        for _ in range(100):
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                # Need to increase sum: add to elements with room
                mask = h < 1.0
                if not np.any(mask):
                    break
                room = 1.0 - h[mask]
                total_room = np.sum(room)
                if total_room <= 0:
                    break
                add = np.minimum(diff, room)
                h[mask] += add
            else:
                # Need to decrease sum: subtract from elements with value
                mask = h > 0.0
                if not np.any(mask):
                    break
                total_val = np.sum(h[mask])
                if total_val <= 0:
                    break
                sub = np.minimum(-diff, h[mask])
                h[mask] -= sub
            diff = N / 2.0 - np.sum(h)
    return h


def construct():
    t0 = time.time()
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _max_overlap(h0)
    
    # Try multiple initializations and optimizers
    initializations = [
        np.full(N, 0.5),
        np.full(N, 0.5) + 0.1 * np.sin(2 * np.pi * np.arange(N) / N),
        np.full(N, 0.5) + 0.1 * np.cos(2 * np.pi * np.arange(N) / N),
        np.full(N, 0.5) + 0.05 * np.sin(4 * np.pi * np.arange(N) / N),
    ]
    
    for h_init in initializations:
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 10:
            break
        
        h_init = _project_to_valid(h_init)
        
        # Try L-BFGS-B with smooth objective
        try:
            res = minimize(
                _objective_smooth,
                h_init,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_candidate = _project_to_valid(res.x)
            val = _max_overlap(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
        
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 10:
            break
        
        # Try SLSQP with smooth objective
        try:
            res2 = minimize(
                _objective_smooth,
                h_init,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            h_candidate = _project_to_valid(res2.x)
            val = _max_overlap(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
    
    # Final polish with direct objective on best result
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 5:
        try:
            res3 = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 50, 'ftol': 1e-15}
            )
            h_candidate = _project_to_valid(res3.x)
            val = _max_overlap(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
    
    return _project_to_valid(best_h)
>>>>>>> REPLACE

