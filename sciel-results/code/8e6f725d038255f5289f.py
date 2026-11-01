<<<<<<< SEARCH
def _slsqp_polish(c0, r0, max_time=30.0):
    """High-precision polish using SLSQP with active constraints."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    def _get_pairs_vectorized(c, r_target, margin=0.005):
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)

    step = 3e-5
    max_iter = 120
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        r_target = best_r + step
        pairs = _get_pairs_vectorized(best_c, r_target, margin=0.006)
        npairs = len(pairs)

        def constraints_func(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c) - r_target
            bd_flat = bd.ravel()
            if npairs == 0:
                return bd_flat
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            pd_vals = dists - 2 * r_target
            return np.concatenate([bd_flat, pd_vals])

        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = 2 * n + npairs
            jac = np.zeros((total_constraints, 2 * n))
            signs = -np.sign(c)
            for i in range(n):
                jac[2*i, 2*i] = signs[i, 0]
                jac[2*i+1, 2*i+1] = signs[i, 1]
            if npairs == 0:
                return jac
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            row_indices = 2 * n + np.arange(npairs)
            col_i = pairs[:, 0] * 2
            col_j = pairs[:, 1] * 2
            np.add.at(jac, (row_indices, col_i), unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_i + 1), unit_vecs[:, 1])
            np.add.at(jac, (row_indices, col_j), -unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_j + 1), -unit_vecs[:, 1])
            return jac

        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
                ],
                options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-13:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.08, 1.5e-4)
            else:
                step *= 0.4
                if step < 5e-8:
                    break
        except Exception:
            break
    return best_c, best_r
=======
def _slsqp_polish(c0, r0, max_time=30.0):
    """High-precision polish using cutting-plane method with active constraints."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    def _get_active_constraints(c, r_target, margin=0.008):
        """Get active pairwise and boundary constraints."""
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        
        # Boundary constraints: circles close to boundary
        boundary_dists = 0.5 - np.abs(c)
        boundary_indices = []
        for i in range(n):
            for k in range(2):
                if boundary_dists[i, k] < r_target + margin:
                    boundary_indices.append((i, k))
        
        return (np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32),
                np.array(boundary_indices, dtype=np.int32) if boundary_indices else np.empty((0, 2), dtype=np.int32))

    # Cutting-plane iteration
    max_cp_iter = 80
    for cp_iter in range(max_cp_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        # Get active constraints at current radius
        pairs, boundary_idx = _get_active_constraints(best_c, best_r, margin=0.01)
        npairs = len(pairs)
        nbound = len(boundary_idx)
        
        # Build constraint matrix for QP: maximize t subject to
        # ||c_i - c_j|| >= 2t for active pairs
        # 0.5 - |c_ik| >= t for active boundary constraints
        # This is non-convex due to the norm, so we linearize around current point
        
        # Linearized constraints:
        # For pair (i,j): (c_i - c_j)·(x_i - x_j) >= ||c_i - c_j|| - 2t
        # For boundary (i,k): -sign(c_ik)*x_ik + 0.5 >= t
        
        # Use scipy's minimize with linear constraints approximation
        # Variables: x (2n coords) + t (1) = 2n+1 variables
        
        def _build_linear_constraints(c_ref, pairs, boundary_idx, r_ref):
            """Build linearized constraint matrix A, b such that A @ [x; t] >= b"""
            total_constraints = npairs + nbound
            if total_constraints == 0:
                return None, None
            
            A = np.zeros((total_constraints, 2 * n + 1))
            b = np.zeros(total_constraints)
            
            # Pairwise constraints (linearized)
            for idx, (i, j) in enumerate(pairs):
                ci = c_ref[i]
                cj = c_ref[j]
                diff = ci - cj
                dist = np.linalg.norm(diff)
                if dist < 1e-14:
                    dist = 1e-14
                    diff = np.array([1e-7, 0])
                unit = diff / dist
                
                # Constraint: (x_i - x_j) · unit >= dist - 2t
                # => unit[0]*x_i0 - unit[0]*x_j0 + unit[1]*x_i1 - unit[1]*x_j1 + 2t >= dist
                row = 2 * idx
                A[row, 2*i] = unit[0]
                A[row, 2*i+1] = unit[1]
                A[row, 2*j] = -unit[0]
                A[row, 2*j+1] = -unit[1]
                A[row, 2*n] = 2.0  # coefficient of t
                b[row] = dist
            
            # Boundary constraints
            for idx, (i, k) in enumerate(boundary_idx):
                row = npairs + idx
                sign = np.sign(c_ref[i, k])
                if sign == 0:
                    sign = 1
                # Constraint: -sign * x_ik + 0.5 >= t
                # => -sign * x_ik - t >= -0.5
                col = 2 * i + k
                A[row, col] = -sign
                A[row, 2*n] = -1.0
                b[row] = -0.5
            
            return A, b
        
        A, b = _build_linear_constraints(best_c, pairs, boundary_idx, best_r)
        
        if A is None:
            # No active constraints, can potentially increase radius
            # Try a small step
            best_r += 1e-6
            continue
        
        # Solve QP: minimize -t (maximize t) subject to A @ [x; t] >= b, bounds on x
        # Convert to standard form: minimize -t subject to -A @ [x; t] <= -b
        
        def objective(xt):
            # xt = [x1, x2, ..., x2n, t]
            return -xt[2*n]  # minimize -t = maximize t
        
        # Bounds: x in [-0.5+eps, 0.5-eps], t in [0, 0.5]
        bounds_x = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (2 * n)
        bounds_t = [(0.0, 0.5)]
        all_bounds = bounds_x + bounds_t
        
        # Inequality constraints: -A @ z <= -b
        def cons_func(z):
            return -b - (-A) @ z  # should be >= 0, i.e., -A@z <= -b
        
        # Actually, let's use the form: A @ z >= b, which is -A @ z <= -b
        # scipy minimize expects fun >= 0 for 'ineq'
        
        constraints = [{'type': 'ineq', 'fun': lambda z: A @ z - b}]
        
        x0 = np.concatenate([best_c.flatten(), [best_r]])
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=all_bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            
            if result.success or result.fun < 0:
                x_new = result.x[:2*n].reshape(-1, 2)
                t_new = result.x[2*n]
                x_new = np.clip(x_new, -0.5 + 1e-10, 0.5 - 1e-10)
                
                # Verify the actual radius
                r_actual = _compute_radius(x_new)
                
                if r_actual > best_r + 1e-12:
                    best_c = x_new
                    best_r = r_actual
                    continue
                elif abs(r_actual - best_r) < 1e-12:
                    # No improvement, check if we're at a local optimum
                    # Try a different margin
                    continue
                else:
                    # The solution decreased radius, keep best
                    continue
            else:
                # Optimization failed, break
                break
        except Exception:
            break
    
    return best_c, best_r
>>>>>>> REPLACE

