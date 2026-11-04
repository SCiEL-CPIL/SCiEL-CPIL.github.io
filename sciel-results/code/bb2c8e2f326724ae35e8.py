"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    Note: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x = v*1 + w*0.5
    y = w*sqrt(3)/2
    """
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = (np.sqrt(3) / 2) * w
    return np.column_stack((x, y))

def _cart_to_bary(points):
    """
    Convert Cartesian points to barycentric coordinates (u, v).
    x = v + 0.5(1-u-v) = 0.5 + 0.5v - 0.5u
    y = (sqrt(3)/2)(1-u-v)
    => 1-u-v = 2y/sqrt(3) => u+v = 1 - 2y/sqrt(3)
    => v = 2x - 1 + u
    Substitute u: u + (2x - 1 + u) = 1 - 2y/sqrt(3)
    2u = 2 - 2x - 2y/sqrt(3)
    u = 1 - x - y/sqrt(3)
    v = 2x - 1 + u = 2x - 1 + 1 - x - y/sqrt(3) = x - y/sqrt(3)
    """
    x = points[:, 0]
    y = points[:, 1]
    u = 1.0 - x - y / np.sqrt(3)
    v = x - y / np.sqrt(3)
    return np.column_stack((u, v))

def _all_triangle_areas_bary(bary):
    """
    Compute all triangle areas using barycentric coordinates.
    Area = 0.5 * |det([u_i, v_i, w_i; u_j, v_j, w_j; u_k, v_k, w_k])| * Area(Triangle)
    Since we are maximizing the ratio, we can just use the determinant magnitude.
    The actual area is det * (sqrt(3)/4).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx, j_idx, k_idx = _get_indices()
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    # Determinant of 3x3 matrix formed by rows Pi, Pj, Pk
    # det = Pi0*(Pj1*Pk2 - Pj2*Pk1) - Pi1*(Pj0*Pk2 - Pj2*Pk0) + Pi2*(Pj0*Pk1 - Pj1*Pk0)
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return 0.5 * np.abs(det) * (np.sqrt(3) / 4)

def _min_triangle_area_bary(bary):
    """Compute the smallest triangle area."""
    return np.min(_all_triangle_areas_bary(bary))

def _soft_min_objective(bary, beta):
    """
    Objective: -soft_min(areas).
    soft_min = -1/beta * log(sum(exp(-beta * areas)))
    So -soft_min = 1/beta * log(sum(exp(-beta * areas)))
    """
    areas = _all_triangle_areas_bary(bary)
    # Shift for numerical stability
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return log_sum_exp / beta

def _soft_min_grad(bary, beta):
    """
    Gradient of -soft_min(areas) w.r.t. barycentric coordinates.
    f(bary) = 1/beta * log(sum(exp(-beta * A_i)))
    df/dbary = 1/beta * (1/sum(exp(-beta A_i))) * sum(exp(-beta A_i) * (-beta) * dA_i/dbary)
             = - sum(w_i * dA_i/dbary)
    where w_i = exp(-beta A_i) / sum(exp(-beta A_j))
    """
    areas = _all_triangle_areas_bary(bary)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx, j_idx, k_idx = _get_indices()
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    # A_i = 0.5 * |det_i| * C
    # dA_i/dbary_m = 0.5 * C * sign(det_i) * d(det_i)/dbary_m
    # Let K = 0.5 * C * sign(det_i)
    sign_det = np.sign(det)
    sign_det[np.abs(det) < 1e-15] = 1.0
    K = 0.5 * (np.sqrt(3) / 4) * sign_det
    
    # Weights w_i
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # d(det)/d(P_m) for m in {i, j, k}
    # det = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    # d(det)/du_i = v_j w_k - w_j v_k
    # d(det)/dv_i = -(u_j w_k - w_j u_k) = w_j u_k - u_j w_k
    # d(det)/dw_i = u_j v_k - v_j u_k
    
    # Similarly for j and k.
    d_det_du_i = Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1]
    d_det_dv_i = Pj[:, 2] * Pk[:, 0] - Pj[:, 0] * Pk[:, 2]
    d_det_dw_i = Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]
    
    d_det_du_j = -(Pi[:, 1] * Pk[:, 2] - Pi[:, 2] * Pk[:, 1])
    d_det_dv_j = Pi[:, 0] * Pk[:, 2] - Pi[:, 2] * Pk[:, 0]
    d_det_dw_j = -(Pi[:, 0] * Pk[:, 1] - Pi[:, 1] * Pk[:, 0])
    
    d_det_du_k = (Pi[:, 1] * Pj[:, 2] - Pi[:, 2] * Pj[:, 1])
    d_det_dv_k = -(Pi[:, 0] * Pj[:, 2] - Pi[:, 2] * Pj[:, 0])
    d_det_dw_k = (Pi[:, 0] * Pj[:, 1] - Pi[:, 1] * Pj[:, 0])
    
    # Accumulate gradients for u, v, w
    # Note: u+v+w=1, so du+dv+dw=0. The gradient in the constrained space is the projection.
    # However, since we parameterize by (u, v) and w is dependent, we can just compute d/du and d/dv.
    # dA/du = K * d_det/du
    # dA/dv = K * d_det/dv
    
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    # Contribution from point i
    w_i = weights
    grad_u[i_idx] += K * w_i * d_det_du_i
    grad_v[i_idx] += K * w_i * d_det_dv_i
    
    # Contribution from point j
    grad_u[j_idx] += K * w_i * d_det_du_j
    grad_v[j_idx] += K * w_i * d_det_dv_j
    
    # Contribution from point k
    grad_u[k_idx] += K * w_i * d_det_du_k
    grad_v[k_idx] += K * w_i * d_det_dw_k # Wait, d_det_dw_k is d(det)/dw_k. 
    # But we want dA/dv_k. 
    # A depends on u_k, v_k, w_k. 
    # dA/dv_k = K * d_det/dv_k.
    # My d_det_dv_k above is correct.
    # Let's re-verify d_det_dw_k usage. 
    # The gradient of the objective function f(bary) w.r.t. the variables u, v is what we need for L-BFGS-B.
    # f = - sum w_i A_i
    # df/du_m = - sum_{i: m in triple} w_i dA_i/du_m
    # df/dv_m = - sum_{i: m in triple} w_i dA_i/dv_m
    
    # My K includes the sign and constant. 
    # dA_i/du_m = K_i * d_det_i/du_m
    # So df/du_m = - sum w_i K_i d_det_i/du_m
    
    # Let's re-calculate cleanly.
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    # For each triple, we add to the gradients of u and v for the 3 points.
    # dA/du = K * d_det/du
    # dA/dv = K * d_det/dv
    
    # Point i
    grad_u[i_idx] -= K * w_i * d_det_du_i
    grad_v[i_idx] -= K * w_i * d_det_dv_i
    
    # Point j
    grad_u[j_idx] -= K * w_i * d_det_du_j
    grad_v[j_idx] -= K * w_i * d_det_dv_j
    
    # Point k
    grad_u[k_idx] -= K * w_i * d_det_du_k
    grad_v[k_idx] -= K * w_i * d_det_dv_k
    
    # The result is the gradient of f = -soft_min.
    # We want to minimize f, so we return this gradient.
    return np.column_stack((grad_u, grad_v))

def _objective_exact(bary):
    """Objective for exact optimization: -min_area."""
    return -_min_triangle_area_bary(bary)

def _project_to_bary_simplex(bary):
    """Project points to the valid barycentric simplex."""
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling in barycentric coords.
    """
    points_bary = np.zeros((n_points, 2))
    points_bary[0] = [1/3, 1/3]
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([r1, r2])
        
        # Distance in barycentric space is proportional to Euclidean distance
        diffs = candidates[:, np.newaxis, :] - points_bary[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points_bary[k] = candidates[best_idx]
        
    return points_bary

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n + 1):
            for j in range(grid_n + 1 - i):
                u = i / grid_n
                v = j / grid_n
                points.append([u, v])
        
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
                points = np.vstack([points, [r1, r2]])
            inits.append(points)
            
    return inits

def _simulated_annealing(bary, rng, max_iter=3000, t_start=0.05, t_end=0.001, deadline=None):
    """
    Simulated annealing in barycentric space.
    """
    current_bary = bary.copy()
    current_score = _min_triangle_area_bary(current_bary)
    best_bary_sa = current_bary.copy()
    best_score_sa = current_score
    
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, max_iter - 1))
    
    for step in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        T = t_start * (cooling_rate ** step)
        
        r = rng.random()
        if r < 0.2:
            # Swap two points
            i, j = rng.choice(N, 2, replace=False)
            new_bary = current_bary.copy()
            new_bary[i], new_bary[j] = new_bary[j].copy(), new_bary[i].copy()
        elif r < 0.5:
            # Small Gaussian perturbation
            noise = rng.normal(0, 0.02, size=(N, 2))
            new_bary = current_bary + noise
        elif r < 0.8:
            # Move one point significantly
            i = rng.randint(N)
            new_bary = current_bary.copy()
            new_bary[i] += rng.normal(0, 0.1, size=2)
        else:
            # Move two points
            i, j = rng.choice(N, 2, replace=False)
            new_bary = current_bary.copy()
            new_bary[i] += rng.normal(0, 0.08, size=2)
            new_bary[j] += rng.normal(0, 0.08, size=2)
            
        new_bary = _project_to_bary_simplex(new_bary)
        new_score = _min_triangle_area_bary(new_bary)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_bary = new_bary
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_bary_sa = current_bary.copy()
                
    return best_bary_sa, best_score_sa

def construct():
    start_time = time.time()
    deadline_global = start_time + 140.0
    
    def time_left():
        return deadline_global - time.time()
        
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Differential Evolution for global search
    # Bounds for u, v: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    if time_left() > 60.0:
        try:
            # Use a smaller population and fewer iterations to fit in time
            de_res = differential_evolution(
                _objective_exact, bounds, 
                maxiter=20, 
                popsize=15, 
                tol=1e-8, 
                seed=42,
                polish=False # We will polish manually
            )
            if de_res.success or de_res.fun < best_score:
                best_score = -de_res.fun
                best_bary = de_res.x.reshape(N, 2)
                best_bary = _project_to_bary_simplex(best_bary)
                # Update score after projection
                best_score = _min_triangle_area_bary(best_bary)
        except Exception:
            pass

    # Phase 2: Multi-start local optimization (L-BFGS-B with soft-min)
    # Generate diverse initializations
    initial_baries = []
    
    if best_bary is not None:
        initial_baries.append(best_bary)
        
    for _ in range(5):
        if time_left() < 40.0:
            break
        bary = _generate_fps_initialization(rng)
        initial_baries.append(bary)
        
    for bary in _structured_inits(rng):
        if time_left() < 40.0:
            break
        initial_baries.append(bary)
        
    betas = [50, 100, 200, 500]
    
    for init_bary in initial_baries:
        if time_left() < 20.0:
            break
            
        current_bary = init_bary.copy()
        
        for beta in betas:
            if time_left() < 10.0:
                break
            try:
                res = minimize(
                    lambda b: _soft_min_objective(b, beta),
                    current_bary,
                    jac=lambda b: _soft_min_grad(b, beta),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_bary = _project_to_bary_simplex(res.x.reshape(N, 2))
                new_score = _min_triangle_area_bary(new_bary)
                
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary
                current_bary = new_bary
            except Exception:
                pass

    # Phase 3: Simulated Annealing from best found
    if best_bary is not None and time_left() > 30.0:
        try:
            sa_bary, sa_score = _simulated_annealing(best_bary, rng, max_iter=2000, deadline=deadline_global - 10.0)
            if sa_score > best_score:
                best_score = sa_score
                best_bary = sa_bary
        except Exception:
            pass

    # Phase 4: Final Nelder-Mead Polish on exact objective
    if best_bary is not None and time_left() > 10.0:
        try:
            # NM works on flat array
            x0 = best_bary.flatten()
            res = minimize(
                lambda x: _objective_exact(x.reshape(N, 2)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            new_bary = _project_to_bary_simplex(res.x.reshape(N, 2))
            new_score = _min_triangle_area_bary(new_bary)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
        except Exception:
            pass

    # Fallback
    if best_bary is None:
        # Grid fallback
        best_bary = np.array([[1/3, 1/3]] * N)
        best_score = _min_triangle_area_bary(best_bary)

    # Convert to Cartesian
    best_points = _bary_to_cart(best_bary[:, 0], best_bary[:, 1])
    
    # Final safety clip
    h = np.sqrt(3) / 2
    for i in range(N):
        xi, yi = best_points[i]
        if yi < 0:
            best_points[i, 1] = 0.0
        if yi > h:
            best_points[i, 1] = h
        if xi < 0:
            best_points[i, 0] = 0.0
        if xi > 1:
            best_points[i, 0] = 1.0
        max_y_left = np.sqrt(3) * xi
        max_y_right = np.sqrt(3) * (1.0 - xi)
        max_y = min(max_y_left, max_y_right)
        if best_points[i, 1] > max_y:
            best_points[i, 1] = max_y
            
    return best_points
