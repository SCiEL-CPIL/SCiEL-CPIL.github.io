<<<<<<< SEARCH
def _slsqp_polish(c0, max_time=20.0):
    """
    Polish using SLSQP with active constraints, but with a limit on constraint count.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    max_outer_iters = 20
    max_constraints = 80  # Limit to keep SLSQP fast
    
    for k in range(max_outer_iters):
        if time.time() > start_time + max_time - 2.0:
            break
        
        pair_indices = _get_active_pairs(c, r, tol_factor=1.03)
        boundary_indices = _get_active_boundary(c, r, tol_factor=1.03)
        
        # If too many constraints, take only the closest ones
        if len(pair_indices) > max_constraints // 2:
            # Sort by distance and take closest
            dist_matrix = squareform(pdist(c, metric='euclidean'))
            pair_dists = [(dist_matrix[i, j], i, j) for i, j in pair_indices]
            pair_dists.sort(key=lambda x: x[0])
            pair_indices = [(i, j) for _, i, j in pair_dists[:max_constraints // 2]]
        
        if len(boundary_indices) > max_constraints // 2:
            boundary_dists = 0.5 - np.abs(c)
            min_b = np.min(boundary_dists, axis=1)
            boundary_indices = list(np.argsort(min_b)[:max_constraints // 2])
        
        if len(pair_indices) == 0 and len(boundary_indices) == 0:
            pair_indices = _get_active_pairs(c, r, tol_factor=1.1)
            boundary_indices = _get_active_boundary(c, r, tol_factor=1.1)
            if len(pair_indices) == 0 and len(boundary_indices) == 0:
                break

        def objective(params):
            return -params[-1]
        
        def objective_grad(params):
            grad = np.zeros_like(params)
            grad[-1] = -1.0
            return grad
            
        constraints = []
        
        for i, j in pair_indices:
            def make_pair_con(i, j):
                def con(params):
                    ci = params[2*i:2*i+2]
                    cj = params[2*j:2*j+2]
                    r_val = params[-1]
                    dist = np.linalg.norm(ci - cj)
                    return dist - 2.0 * r_val
                
                def con_grad(params):
                    ci = params[2*i:2*i+2]
                    cj = params[2*j:2*j+2]
                    diff = ci - cj
                    dist = np.linalg.norm(diff)
                    if dist < 1e-12:
                        grad_c = np.zeros(2*N+1)
                        grad_c[-1] = -2.0
                        return grad_c
                    grad_c = np.zeros(2*N+1)
                    grad_c[2*i:2*i+2] = diff / dist
                    grad_c[2*j:2*j+2] = -diff / dist
                    grad_c[-1] = -2.0
                    return grad_c
                
                return {'type': 'ineq', 'fun': con, 'jac': con_grad}
            
            constraints.append(make_pair_con(i, j))
        
        for k_idx in boundary_indices:
            x, y = c[k_idx]
            dist_left = x + 0.5
            dist_right = 0.5 - x
            dist_bottom = y + 0.5
            dist_top = 0.5 - y
            tol = r * 1.03
            
            if dist_left < tol:
                def make_left_con(k_idx):
                    def con(params):
                        return 0.5 + params[2*k_idx] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx] = 1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_left_con(k_idx))
                
            if dist_right < tol:
                def make_right_con(k_idx):
                    def con(params):
                        return 0.5 - params[2*k_idx] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx] = -1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_right_con(k_idx))
                
            if dist_bottom < tol:
                def make_bottom_con(k_idx):
                    def con(params):
                        return 0.5 + params[2*k_idx+1] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx+1] = 1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_bottom_con(k_idx))
                
            if dist_top < tol:
                def make_top_con(k_idx):
                    def con(params):
                        return 0.5 - params[2*k_idx+1] - params[-1]
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k_idx+1] = -1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_top_con(k_idx))
        
        if not constraints:
            break
            
        x0 = np.zeros(2*N + 1)
        x0[:2*N] = c.flatten()
        x0[-1] = r
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 120, 'ftol': 1e-15, 'disp': False}
            )
            
            new_c = result.x[:2*N].reshape(-1, 2)
            new_c = np.clip(new_c, -0.5, 0.5)
            new_r = _compute_radius(new_c)
            
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
                c = new_c.copy()
                r = new_r
            else:
                break
                
        except Exception:
            break
            
    return best_c, best_r
=======
def _slsqp_polish(c0, max_time=20.0):
    """
    Polish using SLSQP with vectorized active constraints and iterative refinement.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    dim = 2 * N + 1
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    def objective(params):
        return -params[-1]
    
    def objective_grad(params):
        grad = np.zeros(dim)
        grad[-1] = -1.0
        return grad
    
    # Iterative refinement: start with loose tolerance, tighten progressively
    tol_factors = [3.0, 1.5, 1.02, 1.005]
    max_constraints = 100
    
    for tol_factor in tol_factors:
        if time.time() > start_time + max_time - 3.0:
            break
        
        # Get active constraints
        dist_matrix = squareform(pdist(c, metric='euclidean'))
        threshold = 2.0 * r * tol_factor
        i_idx, j_idx = np.where(dist_matrix < threshold)
        mask = i_idx < j_idx
        pair_i = i_idx[mask]
        pair_j = j_idx[mask]
        
        if len(pair_i) > max_constraints // 2:
            pair_dists = dist_matrix[pair_i, pair_j]
            order = np.argsort(pair_dists)
            pair_i = pair_i[order[:max_constraints // 2]]
            pair_j = pair_j[order[:max_constraints // 2]]
        
        # Active boundary constraints
        boundary_dists = 0.5 - np.abs(c)  # shape (N, 2)
        b_threshold = r * tol_factor
        # For each point, check which walls are close
        active_b = []  # list of (point_idx, wall) where wall: 0=left,1=right,2=bottom,3=top
        for k_idx in range(N):
            if boundary_dists[k_idx, 0] < b_threshold:  # left
                active_b.append((k_idx, 0))
            if boundary_dists[k_idx, 1] < b_threshold:  # right
                active_b.append((k_idx, 1))
            if boundary_dists[k_idx, 2] < b_threshold:  # bottom
                active_b.append((k_idx, 2))
            if boundary_dists[k_idx, 3] < b_threshold:  # top
                active_b.append((k_idx, 3))
        
        if len(active_b) > max_constraints // 2:
            # Keep closest
            b_vals = []
            for k_idx, wall in active_b:
                if wall == 0: b_vals.append(boundary_dists[k_idx, 0])
                elif wall == 1: b_vals.append(boundary_dists[k_idx, 1])
                elif wall == 2: b_vals.append(boundary_dists[k_idx, 2])
                else: b_vals.append(boundary_dists[k_idx, 3])
            b_vals = np.array(b_vals)
            order = np.argsort(b_vals)
            active_b = [active_b[idx] for idx in order[:max_constraints // 2]]
        
        if len(pair_i) == 0 and len(active_b) == 0:
            continue
        
        # Build vectorized constraint functions
        n_pairs = len(pair_i)
        n_b = len(active_b)
        
        if n_pairs > 0:
            pi = pair_i.astype(np.int64)
            pj = pair_j.astype(np.int64)
        else:
            pi = np.array([], dtype=np.int64)
            pj = np.array([], dtype=np.int64)
        
        if n_b > 0:
            bi_pts = np.array([p for p, w in active_b], dtype=np.int64)
            bi_walls = np.array([w for p, w in active_b], dtype=np.int64)
        else:
            bi_pts = np.array([], dtype=np.int64)
            bi_walls = np.array([], dtype=np.int64)
        
        def con_fun(params):
            vals = np.empty(n_pairs + n_b)
            if n_pairs > 0:
                ci = params[2*pi:2*pi+2].reshape(-1, 2)
                cj = params[2*pj:2*pj+2].reshape(-1, 2)
                diff = ci - cj
                dists = np.linalg.norm(diff, axis=1)
                vals[:n_pairs] = dists - 2.0 * params[-1]
            if n_b > 0:
                x = params[2*bi_pts]
                y = params[2*bi_pts+1]
                # wall 0=left, 1=right, 2=bottom, 3=top
                wall_vals = np.empty(n_b)
                wall_vals[bi_walls == 0] = 0.5 + x[bi_walls == 0]
                wall_vals[bi_walls == 1] = 0.5 - x[bi_walls == 1]
                wall_vals[bi_walls == 2] = 0.5 + y[bi_walls == 2]
                wall_vals[bi_walls == 3] = 0.5 - y[bi_walls == 3]
                vals[n_pairs:] = wall_vals - params[-1]
            return vals
        
        def con_jac(params):
            J = np.zeros((n_pairs + n_b, dim))
            if n_pairs > 0:
                ci = params[2*pi:2*pi+2].reshape(-1, 2)
                cj = params[2*pj:2*pj+2].reshape(-1, 2)
                diff = ci - cj
                dists = np.linalg.norm(diff, axis=1)
                safe_dists = np.where(dists < 1e-12, 1.0, dists)
                unit = diff / safe_dists[:, np.newaxis]
                # Gradient w.r.t. ci: unit / dist, w.r.t. cj: -unit / dist
                for idx in range(n_pairs):
                    J[idx, 2*pi[idx]:2*pi[idx]+2] = unit[idx] / safe_dists[idx]
                    J[idx, 2*pj[idx]:2*pj[idx]+2] = -unit[idx] / safe_dists[idx]
                J[:n_pairs, -1] = -2.0
            if n_b > 0:
                for idx in range(n_b):
                    pt = bi_pts[idx]
                    wall = bi_walls[idx]
                    if wall == 0:
                        J[n_pairs + idx, 2*pt] = 1.0
                    elif wall == 1:
                        J[n_pairs + idx, 2*pt] = -1.0
                    elif wall == 2:
                        J[n_pairs + idx, 2*pt+1] = 1.0
                    else:
                        J[n_pairs + idx, 2*pt+1] = -1.0
                J[n_pairs:, -1] = -1.0
            return J
        
        constraint = {'type': 'ineq', 'fun': con_fun, 'jac': con_jac}
        
        x0 = np.zeros(dim)
        x0[:2*N] = c.flatten()
        x0[-1] = r
        
        maxiter_slsqp = 200 if tol_factor >= 1.5 else 300
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=[constraint],
                options={'maxiter': maxiter_slsqp, 'ftol': 1e-15, 'disp': False}
            )
            
            new_c = result.x[:2*N].reshape(-1, 2)
            new_c = np.clip(new_c, -0.5, 0.5)
            new_r = _compute_radius(new_c)
            
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
                c = new_c.copy()
                r = new_r
            elif tol_factor > 1.02:
                # Still continue to next tighter tolerance
                continue
            else:
                break
                
        except Exception:
            break
        
    return best_c, best_r
>>>>>>> REPLACE

