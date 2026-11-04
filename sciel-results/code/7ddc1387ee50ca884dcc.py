"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_I = None
_IDX_J = None
_IDX_K = None
_N_TRIPLES = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K, _N_TRIPLES
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
        _N_TRIPLES = len(_IDX_I)
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    """
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _scp_solve(params, active_indices, max_lp_iter=50):
    """
    Sequential Convex Programming: solve LP to maximize t subject to linearized area constraints
    for the active set of smallest triangles.
    
    Args:
        params: current parameters (2N,)
        active_indices: indices of triangles to include in active set
        max_lp_iter: maximum number of LP iterations
        
    Returns:
        new_params: improved parameters
        new_active: updated active set indices
    """
    current_params = params.copy()
    current_active = active_indices.copy()
    
    for lp_iter in range(max_lp_iter):
        if time.time() > deadline_global - 5.0:
            break
            
        points = _param_to_points(current_params)
        areas = _all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Update active set: include triangles with area close to minimum
        threshold = min_area * 1.05  # 5% above minimum
        new_active = np.where(areas <= threshold)[0]
        
        # If active set hasn't changed much, we may have converged
        if lp_iter > 5 and len(new_active) == len(current_active) and np.allclose(np.sort(new_active), np.sort(current_active)):
            break
        
        current_active = new_active
        
        if len(current_active) == 0:
            break
        
        # Build LP: maximize t subject to area_i >= t for i in active set
        # Variables: [t, x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}]
        # But we use barycentric parameters: [t, s_0, t_0, s_1, t_1, ...]
        
        n_vars = 1 + 2 * N  # t + 2N parameters
        n_constraints = len(current_active)
        
        # Objective: maximize t => minimize -t
        c = np.zeros(n_vars)
        c[0] = -1.0
        
        # Constraints: area_i(params) >= t for each active triangle
        # Linearize area_i around current_params
        # area_i = 0.5 * |cross_i|
        # We need: 0.5 * |cross_i(params)| - t >= 0
        
        # For linearization, we use the sign of cross at current solution
        i_idx, j_idx, k_idx = _get_indices()
        
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        for idx, tri_idx in enumerate(current_active):
            ii = i_idx[tri_idx]
            jj = j_idx[tri_idx]
            kk = k_idx[tri_idx]
            
            pi = points[ii]
            pj = points[jj]
            pk = points[kk]
            
            cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
            sign = np.sign(cross) if abs(cross) > 1e-15 else 1.0
            
            # Linearize: 0.5 * sign * (cross_linearized) - t >= 0
            # => t - 0.5 * sign * cross_linearized <= 0
            
            # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
            # Linearize around current point:
            # cross_lin = cross_0 + d_cross/dx_i * dx_i + d_cross/dy_i * dy_i + ...
            
            # Derivatives of cross w.r.t. each point:
            # d_cross/dx_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
            # d_cross/dy_i = -(x_j - x_i) + (x_k - x_i) = x_k - x_j
            # d_cross/dx_j = y_k - y_i
            # d_cross/dy_j = -(x_k - x_i)
            # d_cross/dx_k = -(y_j - y_i)
            # d_cross/dy_k = x_j - x_i
            
            d_cross = np.zeros(2 * N)
            d_cross[ii * 2] = points[jj, 1] - points[kk, 1]
            d_cross[ii * 2 + 1] = points[kk, 0] - points[jj, 0]
            d_cross[jj * 2] = points[kk, 1] - points[ii, 1]
            d_cross[jj * 2 + 1] = -(points[kk, 0] - points[ii, 0])
            d_cross[kk * 2] = -(points[jj, 1] - points[ii, 1])
            d_cross[kk * 2 + 1] = points[jj, 0] - points[ii, 0]
            
            # Now map from Cartesian derivatives to parameter derivatives
            # x = s * (1-t) * V1[0] + s * t * V2[0]
            # y = s * (1-t) * V1[1] + s * t * V2[1]
            
            s_curr = current_params[0::2]
            t_curr = current_params[1::2]
            s_curr = np.clip(s_curr, 1e-10, 1.0 - 1e-10)
            t_curr = np.clip(t_curr, 1e-10, 1.0 - 1e-10)
            
            dx_ds = (1 - t_curr) * V1[0] + t_curr * V2[0]
            dx_dt = -s_curr * V1[0] + s_curr * V2[0]
            dy_ds = (1 - t_curr) * V1[1] + t_curr * V2[1]
            dy_dt = -s_curr * V1[1] + s_curr * V2[1]
            
            # Parameter derivatives:
            # d_cross/ds_i = d_cross/dx_i * dx_ds[i] + d_cross/dy_i * dy_ds[i]
            # d_cross/dt_i = d_cross/dx_i * dx_dt[i] + d_cross/dy_i * dy_dt[i]
            
            d_param = np.zeros(2 * N)
            d_param[0::2] = d_cross[0::2] * dx_ds + d_cross[1::2] * dy_ds
            d_param[1::2] = d_cross[0::2] * dx_dt + d_cross[1::2] * dy_dt
            
            # Constraint: t - 0.5 * sign * (cross_0 + d_param · (params - params_0)) <= 0
            # => t - 0.5 * sign * d_param · params <= -0.5 * sign * cross_0 + 0.5 * sign * d_param · params_0
            
            cross_0 = cross
            rhs = -0.5 * sign * cross_0 + 0.5 * sign * np.dot(d_param, current_params)
            
            # A_ub[idx] corresponds to: t - 0.5 * sign * d_param · params <= rhs
            # Variable order: [t, s_0, t_0, s_1, t_1, ...]
            A_ub[idx, 0] = 1.0
            A_ub[idx, 1:] = -0.5 * sign * d_param
            b_ub[idx] = rhs
        
        # Bounds: t >= 0, 0 <= s_i, t_i <= 1
        bounds_lp = [(0, None)] + [(0, 1)] * (2 * N)
        
        try:
            res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
            
            if res_lp.success:
                new_params = res_lp.x[1:]
                new_params = np.clip(new_params, 0.0, 1.0)
                
                # Check if this improved the minimum area
                new_points = _param_to_points(new_params)
                new_areas = _all_triangle_areas(new_points)
                new_min_area = np.min(new_areas)
                
                if new_min_area > min_area + 1e-10:
                    current_params = new_params
                else:
                    # No improvement, break
                    break
            else:
                break
        except Exception:
            break
    
    # Final active set update
    points = _param_to_points(current_params)
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    threshold = min_area * 1.05
    final_active = np.where(areas <= threshold)[0]
    
    return current_params, final_active

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # Symmetric patterns: 1 center + 10 around
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def _simulated_annealing(params, rng, max_iter=5000, t_start=0.1, t_end=0.001):
    """
    Simulated annealing to explore the parameter space.
    """
    current_params = params.copy()
    current_score = -_objective(current_params)
    best_params_sa = current_params.copy()
    best_score_sa = current_score
    
    n_steps = max_iter
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, n_steps - 1))
    
    for step in range(n_steps):
        if time.time() > deadline_global:
            break
        
        T = t_start * (cooling_rate ** step)
        
        # Generate neighbor: mix of jitter and swap
        r = rng.random()
        if r < 0.3:
            # Topological swap: swap barycentric coords of two random points
            i, j = rng.choice(N, 2, replace=False)
            new_params = current_params.copy()
            # Swap s and t for points i and j
            new_params[i*2], new_params[j*2] = new_params[j*2], new_params[i*2]
            new_params[i*2+1], new_params[j*2+1] = new_params[j*2+1], new_params[i*2+1]
        elif r < 0.7:
            # Local jitter: Gaussian perturbation
            noise = rng.normal(0, 0.05, size=2*N)
            new_params = np.clip(current_params + noise, 0.0, 1.0)
        else:
            # Larger move: perturb one point significantly
            i = rng.randint(N)
            new_params = current_params.copy()
            new_params[i*2] = np.clip(current_params[i*2] + rng.normal(0, 0.2), 0.0, 1.0)
            new_params[i*2+1] = np.clip(current_params[i*2+1] + rng.normal(0, 0.2), 0.0, 1.0)
        
        new_score = -_objective(new_params)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_params = new_params
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_params_sa = current_params.copy()
    
    return best_params_sa, best_score_sa

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 140.0
    
    def time_left():
        return deadline_global - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 10
    for i in range(n_fps_inits):
        if time_left() < 40.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add structured inits
    for pts in _structured_inits(rng):
        if time_left() < 40.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    # Add random initializations
    for i in range(10):
        if time_left() < 40.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP
    for start_p in initial_params:
        if time_left() < 20.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Simulated annealing to escape local minima
    if best_params is not None and time_left() > 50.0:
        try:
            sa_params, sa_score = _simulated_annealing(best_params, rng, max_iter=3000)
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_params.copy()
        except Exception:
            pass

    # 4. NEW: Sequential Convex Programming (SCP) with LP
    if best_params is not None and time_left() > 40.0:
        try:
            # Get initial active set
            points = _param_to_points(best_params)
            areas = _all_triangle_areas(points)
            min_area = np.min(areas)
            threshold = min_area * 1.05
            active_set = np.where(areas <= threshold)[0]
            
            # Run SCP
            scp_params, _ = _scp_solve(best_params, active_set, max_lp_iter=30)
            
            # Evaluate
            scp_score = -_objective(scp_params)
            if scp_score > best_score:
                best_score = scp_score
                best_params = scp_params.copy()
        except Exception:
            pass

    # 5. Polish the best solution found with Nelder-Mead
    if best_params is not None and time_left() > 30.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 6. Try a few more SLSQP runs from perturbed best solution
    if best_params is not None and time_left() > 25.0:
        for i in range(5):
            if time_left() < 15.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    # 7. Run SCP again on the best solution for further refinement
    if best_params is not None and time_left() > 20.0:
        try:
            points = _param_to_points(best_params)
            areas = _all_triangle_areas(points)
            min_area = np.min(areas)
            threshold = min_area * 1.02  # Tighter threshold for refinement
            active_set = np.where(areas <= threshold)[0]
            
            scp_params, _ = _scp_solve(best_params, active_set, max_lp_iter=20)
            
            scp_score = -_objective(scp_params)
            if scp_score > best_score:
                best_score = scp_score
                best_params = scp_params.copy()
        except Exception:
            pass

    # 8. Final high-precision polish with SLSQP then Nelder-Mead
    if best_params is not None and time_left() > 15.0:
        # First SLSQP polish
        try:
            res = minimize(
                _objective, best_params, method='SLSQP', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'eps': 1e-12}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Then Nelder-Mead with very tight tolerances
        if time_left() > 10.0:
            try:
                res = minimize(
                    _objective, best_params, method='Nelder-Mead',
                    options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback: Centered grid
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return fallback_points

    # Convert best params to points
    best_points = _param_to_points(best_params)
    
    # Validate all points are inside the triangle (with small tolerance)
    h = np.sqrt(3) / 2
    for i in range(N):
        x, y = best_points[i]
        # Check barycentric coordinates
        lam2 = 2.0 * y / np.sqrt(3)
        lam1 = x - 0.5 * lam2
        lam3 = 1.0 - lam1 - lam2
        # If any coordinate is significantly negative, project to nearest point
        if lam1 < -1e-8 or lam2 < -1e-8 or lam3 < -1e-8:
            # Project to triangle: clip barycentric coords
            lams = np.array([lam1, lam2, lam3])
            lams = np.maximum(lams, 0.0)
            lams = lams / np.sum(lams)
            best_points[i] = lams[0] * V0 + lams[1] * V1 + lams[2] * V2
    
    return best_points
