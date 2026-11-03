<<<<<<< SEARCH
def _high_precision_polish(c, r, t0, time_budget=15.0):
    """
    Very fine-grained local search with adaptive step sizes and
    multiple random restarts of the search direction.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Start with fine steps, then try even finer
    for step_scale in [5e-6, 2e-6, 1e-6, 5e-7]:
        if (time.time() - t0) > t0 + time_budget:
            break
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 2], [-1, 2], [1, -2], [-1, -2],
            [2, 1], [-2, 1], [2, -1], [-2, -1]
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        for _ in range(50):
            if (time.time() - t0) > t0 + time_budget:
                break
            improved = False
            for i in range(N):
                for d in directions:
                    c_try = c.copy()
                    c_try[i] += step_scale * d
                    c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                    r_try = _radius_fast(c_try)
                    if r_try > r_cur + 1e-13:
                        c = c_try
                        r_cur = r_try
                        improved = True
            if not improved:
                break
    
    return c, r_cur
=======
def _high_precision_polish(c, r, t0, time_budget=15.0):
    """
    Very fine-grained local search with adaptive step sizes and
    multiple random restarts of the search direction.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Start with fine steps, then try even finer
    for step_scale in [5e-6, 2e-6, 1e-6, 5e-7]:
        if (time.time() - t0) > t0 + time_budget:
            break
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 2], [-1, 2], [1, -2], [-1, -2],
            [2, 1], [-2, 1], [2, -1], [-2, -1]
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        for _ in range(50):
            if (time.time() - t0) > t0 + time_budget:
                break
            improved = False
            for i in range(N):
                for d in directions:
                    c_try = c.copy()
                    c_try[i] += step_scale * d
                    c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                    r_try = _radius_fast(c_try)
                    if r_try > r_cur + 1e-13:
                        c = c_try
                        r_cur = r_try
                        improved = True
            if not improved:
                break
    
    return c, r_cur


def _active_set_sqp(c, r, t0, time_budget=10.0, max_iter=20):
    """
    Active-set SQP refinement.
    Identifies the 'kissing' graph (active contacts and walls) and solves
    a linear least-squares problem to find a displacement that increases r
    while keeping active constraints satisfied.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    for _ in range(max_iter):
        if (time.time() - t0) > t0 + time_budget:
            break
            
        # Identify active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        # Tolerance for "active"
        tol = 1e-5 * r_cur
        
        active_pairs = np.argwhere(d < 2 * r_cur + tol)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i, k] < r_cur + tol:
                    active_walls.append((i, k))
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
            
        # Build the Jacobian of the constraints w.r.t. positions
        # We want to solve for dx such that:
        # 1. Active pair distances increase by 2*dr
        # 2. Active wall distances increase by dr
        # 3. We maximize dr (or minimize -dr)
        
        # Number of variables: 2*N
        # Number of constraints: len(active_pairs) + len(active_walls)
        
        # If constraints > variables, we have a rigid or over-constrained system.
        # We use least squares to find the best direction.
        
        A = []
        b = []
        
        for i, j in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12: continue
            # Gradient of dist(i,j) w.r.t. ci is dij/norm, w.r.t. cj is -dij/norm
            # We want dist(i,j) + 2*dr. 
            # d(dist)/dt = (dij/norm) . (dx_i - dx_j)
            # We want (dij/norm) . (dx_i - dx_j) = 2 * dr
            # This is a constraint on dx.
            # Row in A: [0..0, dij/norm, 0..0, -dij/norm, 0..0]
            # RHS: 2 * dr. But dr is also a variable?
            # Actually, we can set up the system to maximize r directly.
            # Let's use a simpler heuristic: push apart active pairs and away from walls.
            
            # Force direction for i: dij / norm
            # Force direction for j: -dij / norm
            row = np.zeros(2 * N)
            row[2*i:2*i+2] = dij / norm
            row[2*j:2*j+2] = -dij / norm
            A.append(row)
            # We want to increase the distance, so we want this dot product to be positive.
            # In a least squares sense, we can't easily mix dr.
            # Let's just push apart with a small step.
            b.append(1.0) # Target increase
            
        for i, k in active_walls:
            # Wall constraint: 0.5 - |c_ik| >= r
            # Gradient w.r.t. c_ik is -sign(c_ik)
            # We want to increase 0.5 - |c_ik|, so move c_ik towards center.
            # Direction: -sign(c_ik)
            row = np.zeros(2 * N)
            sign = np.sign(c[i, k])
            if sign == 0: sign = 1.0
            row[2*i + k] = -sign
            A.append(row)
            b.append(1.0)
            
        if not A:
            break
            
        A = np.array(A)
        b = np.array(b)
        
        # Solve A dx = b using least squares
        # This gives a direction that tries to satisfy all active constraints increasing.
        # If the system is rigid, the solution will be small or zero.
        dx, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Normalize the displacement to a small step
        norm_dx = np.linalg.norm(dx)
        if norm_dx < 1e-12:
            break
            
        dx = dx / norm_dx
        
        # Try a small step in this direction
        # We need to ensure we don't violate inactive constraints.
        # We'll use a line search or just a fixed small step.
        step = 1e-5
        for _ in range(10):
            c_try = c + step * dx
            c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
            r_try = _radius_fast(c_try)
            if r_try > r_cur + 1e-14:
                c = c_try
                r_cur = r_try
                break
            step *= 0.5
        else:
            # If no improvement with smallest step, try opposite? Or break?
            # If we can't improve in the "push apart" direction, we might be at a local max
            # or the active set is rigid.
            break
            
    return c, r_cur
>>>>>>> REPLACE

