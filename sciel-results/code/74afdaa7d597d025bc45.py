<<<<<<< SEARCH
def _expansion_relaxation_polish(x, y, r, w, h, max_iter=2000, tol=1e-9):
    """Iteratively expand circles and relax positions to remove overlaps."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for iteration in range(max_iter):
        # Try to expand each circle
        for i in range(N):
            # Max possible radius given walls
            r_max_wall = min(x[i], w - x[i], y[i], h - y[i])
            
            # Max radius given other circles
            r_max_other = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    r_max_other = min(r_max_other, dist - r[j])
            
            r_target = min(r_max_wall, r_max_other, r[i] + 0.001)
            r_target = max(r_target, r[i] * 0.9)  # Don't shrink too much
            
            if r_target > r[i] + tol:
                r[i] = r_target
            
            # Relax position to resolve any remaining overlaps
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    x[i] += nx * overlap * 0.5
                    y[i] += ny * overlap * 0.5
                    x[j] -= nx * overlap * 0.5
                    y[j] -= ny * overlap * 0.5
            
            # Clamp to walls
            r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    return x, y, r
=======
def _expansion_relaxation_polish(x, y, r, w, h, max_iter=2000, tol=1e-9):
    """Vectorized iterative expansion and relaxation."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for iteration in range(max_iter):
        # Vectorized: compute max radius for each circle given walls
        r_max_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        
        # Vectorized: compute max radius given other circles
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(d_sq, 0.0))
        np.fill_diagonal(dist, np.inf)
        
        # For circle i, max radius = min over j of (dist[i,j] - r[j])
        r_max_other = np.min(dist - r[None, :], axis=1)
        
        r_target = np.minimum(np.minimum(r_max_wall, r_max_other), r + 0.002)
        r_target = np.maximum(r_target, r * 0.9)
        
        # Only expand if beneficial
        r_target = np.where(r_target > r + tol, r_target, r)
        r = r_target
        
        # Vectorized push-apart for overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-20))
        np.fill_diagonal(dist, 1.0)
        
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(r_sum - dist, 0.0)
        
        # Normalize direction
        nx = dx / dist
        ny = dy / dist
        
        # Accumulate forces (each circle gets half the push from each overlap)
        fx = np.sum(nx * overlap * 0.5, axis=1)
        fy = np.sum(ny * overlap * 0.5, axis=1)
        
        x += fx
        y += fy
        
        # Clamp to walls
        r = np.minimum(r, np.minimum(np.minimum(x, w - x), np.minimum(y, h - y)))
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        # Check convergence
        if np.max(np.abs(fx)) + np.max(np.abs(fy)) < tol:
            break
    
    return x, y, r

def _epigraph_polish(x, y, r, w, h, max_iter=5000, tol=1e-12):
    """High-precision final polish using smooth objective minimization."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Use SLSQP with very tight tolerance on a small neighborhood
    z = np.zeros(3 * N + 1)
    z[0::3][:N] = x
    z[1::3][:N] = y
    z[2::3][:N] = r
    z[-1] = w
    
    # Tight bounds around current solution
    bounds = []
    for i in range(N):
        bounds.append((max(0.001, x[i] - 0.05), min(w - 0.001, x[i] + 0.05)))
        bounds.append((max(0.001, y[i] - 0.05), min(h - 0.001, y[i] + 0.05)))
        bounds.append((max(0.001, r[i] * 0.5), r[i] * 1.1))
    bounds.append((max(0.3, w - 0.1), min(1.7, w + 0.1)))
    
    try:
        res = minimize(
            _vectorized_objective,
            z,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
        )
        
        if res.fun < _vectorized_objective(z) - 1e-10:
            z_res = res.x
            x_new = z_res[0::3][:N]
            y_new = z_res[1::3][:N]
            r_new = z_res[2::3][:N]
            w_new = z_res[-1]
            h_new = 2.0 - w_new
            
            if _is_valid(x_new, y_new, r_new, w_new, h_new):
                # Verify sum improved
                if np.sum(r_new) > np.sum(r) - 1e-10:
                    return x_new, y_new, r_new, w_new, h_new
    except Exception:
        pass
    
    return x, y, r, w, h
>>>>>>> REPLACE

