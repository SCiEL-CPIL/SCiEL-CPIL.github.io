<<<<<<< SEARCH
def _min_enclosing_hex_side_lp(inner_hexes, angle_deg, all_vertices=None):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # For each of 6 directions, compute normals
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Vectorized constraint construction
    rhs = -(all_vertices @ normals.T)  # (n_pts, 6)
    b_ub = rhs.T.flatten()  # (6 * n_pts,)
    
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], n_pts),
        np.repeat(-normals[:, 1], n_pts),
        np.full(6 * n_pts, -1.0)
    ])
    
    # Add r >= 0 constraint
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs = centered @ normals.T
        r_opt = projs.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg
=======
def _min_enclosing_hex_side_smooth(inner_hexes, all_vertices=None, beta=100.0):
    """
    Find optimal center and side length using smooth log-sum-exp approximation.
    Optimizes (cx, cy, theta) simultaneously.
    """
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Precompute vertices as (N, 2)
    verts = all_vertices
    
    def objective(params):
        cx, cy, theta_rad = params
        # Normals for the 6 faces of the outer hexagon
        # Face k normal is at angle theta + 30 + 60*k
        angles = theta_rad + math.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
        
        # Projections of all vertices onto all 6 normals
        # diff (N, 2), normals.T (2, 6) -> projs (N, 6)
        diffs = verts - np.array([cx, cy])
        projs = diffs @ normals.T  # (N, 6)
        
        # We want to minimize r such that projs <= r
        # Smooth max approximation: (1/beta) * log(sum(exp(beta * x)))
        # To avoid overflow, use log-sum-exp trick: max_x + log(sum(exp(x - max_x)))
        # Here we have N*6 values.
        # max_val = max(projs)
        # smooth_max = max_val + (1/beta) * log(sum(exp(beta * (projs - max_val))))
        
        max_val = np.max(projs)
        # If beta is large, exp(beta * (projs - max_val)) is safe because projs - max_val <= 0
        # exp(0) = 1, exp(negative) < 1.
        # Sum is at least 1.
        # log(sum) is stable.
        
        # Handle case where beta is very large and numerical issues arise
        # But with projs - max_val <= 0, it's safe.
        
        smooth_r = max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * (projs - max_val))))
        
        side = smooth_r / (math.sqrt(3) / 2)
        return side
    
    # Initial guess
    cx0 = np.mean(verts[:, 0])
    cy0 = np.mean(verts[:, 1])
    theta0 = 0.0
    
    # Use L-BFGS-B or Nelder-Mead
    # L-BFGS-B is good for smooth functions
    from scipy.optimize import minimize
    
    try:
        # We can try a few starting angles to avoid local minima, 
        # but the function is convex in (cx, cy) for fixed theta, 
        # and periodic in theta. 
        # The global minimum is likely unique or few.
        # Let's run from a few initial theta values.
        
        best_side = float('inf')
        best_cx, best_cy, best_theta = cx0, cy0, theta0
        
        for theta_init in [0.0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]:
            x0 = np.array([cx0, cy0, theta_init])
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_side:
                best_side = res.fun
                best_cx, best_cy, best_theta = res.x
        
        # Final polish with higher beta if time permits
        # The smooth approximation converges to the exact solution as beta -> inf.
        # We can re-run with a higher beta using the previous result as start.
        global _DEADLINE
        if time.time() < _DEADLINE - 10.0:
            def objective_high_beta(params):
                # Re-define with higher beta
                cx, cy, theta_rad = params
                angles = theta_rad + math.radians(30 + 60 * np.arange(6))
                normals = np.column_stack([np.cos(angles), np.sin(angles)])
                diffs = verts - np.array([cx, cy])
                projs = diffs @ normals.T
                max_val = np.max(projs)
                smooth_r = max_val + (1.0 / 1000.0) * np.log(np.sum(np.exp(1000.0 * (projs - max_val))))
                side = smooth_r / (math.sqrt(3) / 2)
                return side
            
            x0 = np.array([best_cx, best_cy, best_theta])
            res2 = minimize(objective_high_beta, x0, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res2.fun < best_side:
                best_side = res2.fun
                best_cx, best_cy, best_theta = res2.x
        
        # Verify with exact max
        angles = best_theta + math.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        diffs = verts - np.array([best_cx, best_cy])
        projs = diffs @ normals.T
        r_exact = np.max(projs)
        side_exact = r_exact / (math.sqrt(3) / 2)
        
        # If the smooth solution is slightly infeasible or just an approximation, 
        # use the exact side length for the reported score, but the center/angle 
        # from the optimizer are good.
        # However, if we report side_exact, we must ensure the hexagon actually fits.
        # Since r_exact is the max projection, a hexagon with side side_exact 
        # centered at (best_cx, best_cy) with angle best_theta WILL contain the points.
        
        return best_cx, best_cy, side_exact, math.degrees(best_theta) % 60.0
        
    except Exception:
        # Fallback to simple bounding box logic
        cx_mean = np.mean(verts[:, 0])
        cy_mean = np.mean(verts[:, 1])
        # Rough estimate
        max_dist = np.max(np.sqrt((verts[:, 0] - cx_mean)**2 + (verts[:, 1] - cy_mean)**2))
        side_est = (max_dist + 1.0) / (math.sqrt(3) / 2)
        return cx_mean, cy_mean, side_est, 0.0
>>>>>>> REPLACE

Now update `_min_enclosing_hex_side` to call the new smooth function.

