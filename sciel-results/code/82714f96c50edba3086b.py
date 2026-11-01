def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    N = 300
    r_target = 0.0531542991469 * 0.9999 

    def get_hex_lattice():
        # Estimate spacing
        # s ~ sqrt(2*pi / (N*sqrt(3)))
        s_est = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        for s in np.linspace(s_est * 0.95, s_est * 1.05, 20):
            k = int(1.0 / s) + 2
            # Generate hex lattice points
            # i, j from -k to k
            # x = (i + 0.5 * (j % 2)) * s
            # y = j * s * sqrt(3) / 2
            # Note: j % 2 in Python for negative j: -1 % 2 = 1. -2 % 2 = 0.
            # This creates a consistent offset pattern.
            
            i_vals = np.arange(-k, k + 1)
            j_vals = np.arange(-k, k + 1)
            I, J = np.meshgrid(i_vals, j_vals, indexing='ij')
            
            X = (I + 0.5 * (J % 2)) * s
            Y = J * s * np.sqrt(3.0) / 2.0
            
            points = np.column_stack((X.ravel(), Y.ravel()))
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0
            points = points[mask]
            
            if len(points) >= N:
                # Sort by distance to center to pick the most central ones
                idx = np.argsort(radius)
                points = points[idx[:N]]
                return points
        
        # Fallback: Random points
        # Generate many random points and pick N that are well-separated?
        # Or just random.
        return np.random.rand(N, 2) * 2 - 1

    init_points = get_hex_lattice()
    
    # Ensure init_points is valid
    if init_points is None or init_points.shape != (N, 2) or not np.all(np.isfinite(init_points)):
        init_points = np.random.rand(N, 2) * 2 - 1
        init_points *= 0.8

    x0 = init_points.flatten()

    def penalty(x):
        c = x.reshape(N, 2)
        if not np.all(np.isfinite(c)):
            return 1e10
        dist_center = np.linalg.norm(c, axis=1)
        boundary_violation = np.maximum(0, dist_center - (1.0 - r_target))
        penalty_boundary = np.sum(boundary_violation**2)
        
        dx = c[:, np.newaxis, 0] - c[np.newaxis, :, 0]
        dy = c[:, np.newaxis, 1] - c[np.newaxis, :, 1]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        overlap_violation = np.maximum(0, 2 * r_target - d)
        penalty_overlap = np.sum(overlap_violation**2)
        
        total = penalty_boundary + penalty_overlap
        if not np.isfinite(total):
            return 1e10
        return total

    def gradient(x):
        c = x.reshape(N, 2)
        if not np.all(np.isfinite(c)):
            return np.ones_like(x)
            
        grad = np.zeros_like(c)
        
        dist_center = np.linalg.norm(c, axis=1)
        mask_b = dist_center > (1.0 - r_target)
        if np.any(mask_b):
            # Avoid division by zero
            safe_dist = np.where(dist_center[mask_b] > 1e-10, dist_center[mask_b], 1e-10)
            coeff = 2 * (dist_center[mask_b] - (1.0 - r_target)) / safe_dist
            grad[mask_b] += coeff[:, np.newaxis] * c[mask_b]
            
        dx = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(dx, axis=2)
        np.fill_diagonal(d, 1.0)
        
        mask_o = d < 2 * r_target
        if np.any(mask_o):
            force_mag = np.zeros_like(d)
            # Avoid division by zero
            safe_d = np.where(d[mask_o] > 1e-10, d[mask_o], 1e-10)
            force_mag[mask_o] = 2 * (2 * r_target - d[mask_o]) / safe_d
            grad += np.sum(force_mag[:, :, np.newaxis] * dx, axis=1)
            
        g_flat = grad.flatten()
        if not np.all(np.isfinite(g_flat)):
            g_flat = np.nan_to_num(g_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        return g_flat

    def time_callback(xk):
        if time.time() - start_time > time_limit - 5:
            raise Exception("Time limit exceeded")

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_x = x0
    best_val = penalty(x0)
    
    try:
        res = minimize(penalty, x0, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8},
                       callback=time_callback)
        if res.fun < best_val and np.all(np.isfinite(res.x)):
            best_x = res.x
            best_val = res.fun
    except Exception as e:
        # If it fails, try a shorter run or just return current best
        pass

    centers = best_x.reshape(N, 2)
    
    # Final cleanup
    dist = np.linalg.norm(centers, axis=1)
    mask = dist > 1.0
    if np.any(mask):
        centers[mask] /= dist[mask, np.newaxis]
        
    return centers
