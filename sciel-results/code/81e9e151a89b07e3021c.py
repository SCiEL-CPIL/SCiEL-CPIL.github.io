"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples to speed up area calculation
# i < j < k
_IDX_I = None
_IDX_J = None
_IDX_K = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    For each point i, we have params s_i, t_i in [0, 1].
    Barycentric coordinates (relative to V0, V1, V2) are:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    Point = lambda_1 * V1 + lambda_2 * V2 + lambda_3 * V0
    Since V0 is (0,0), Point = lambda_1 * V1 + lambda_2 * V2
    """
    s = params[0::2]
    t = params[1::2]
    
    # Ensure bounds are respected
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    # Cartesian coordinates
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    Point = lam1 * V1 + lam2 * V2
    x = lam1 + 0.5 * lam2
    y = (sqrt(3)/2) * lam2
    => lam2 = 2 * y / sqrt(3)
    => lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = lam2 / s  (if s > 0)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    # Clip to valid range [0, 1]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _grad_min_area_params(params):
    """
    Compute the gradient of the negative minimum triangle area w.r.t. params (s, t).
    """
    points = _param_to_points(params)
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    tol = 1e-8
    active = areas <= min_area + tol
    active_indices = np.where(active)[0]
    
    grad_xy = np.zeros((N, 2))
    
    for idx in active_indices:
        i, j, k = i_idx[idx], j_idx[idx], k_idx[idx]
        pi_vec = points[i]
        pj_vec = points[j]
        pk_vec = points[k]
        
        c = cross[idx]
        sign = 1.0 if c > 0 else -1.0
        
        # Gradients of Area w.r.t. Cartesian coordinates
        # dA/dx_i = 0.5 * sign * (y_j - y_k)
        # dA/dy_i = 0.5 * sign * (2*x_i - x_j - x_k)
        # dA/dx_j = 0.5 * sign * (y_k - y_i)
        # dA/dy_j = 0.5 * sign * (x_i - x_k)
        # dA/dx_k = 0.5 * sign * (y_j - y_i)
        # dA/dy_k = 0.5 * sign * (x_j - x_i)
        
        xi, yi = pi_vec
        xj, yj = pj_vec
        xk, yk = pk_vec
        
        # Point i
        grad_xy[i, 0] += 0.5 * sign * (yj - yk)
        grad_xy[i, 1] += 0.5 * sign * (2*xi - xj - xk)
        
        # Point j
        grad_xy[j, 0] += 0.5 * sign * (yk - yi)
        grad_xy[j, 1] += 0.5 * sign * (xi - xk)
        
        # Point k
        grad_xy[k, 0] += 0.5 * sign * (yj - yi)
        grad_xy[k, 1] += 0.5 * sign * (xj - xi)
        
    # Average the gradients of active triangles to get a smooth-ish gradient
    if len(active_indices) > 0:
        grad_xy /= len(active_indices)
        
    # Map grad_xy to grad_params
    s = params[0::2]
    t = params[1::2]
    
    grad_params = np.zeros(2 * N)
    
    for i in range(N):
        si = s[i]
        ti = t[i]
        
        dx_ds = 1.0 - 0.5 * ti
        dx_dt = -0.5 * si
        dy_ds = (np.sqrt(3) / 2) * ti
        dy_dt = (np.sqrt(3) / 2) * si
        
        gx = grad_xy[i, 0]
        gy = grad_xy[i, 1]
        
        grad_params[2*i] = gx * dx_ds + gy * dy_ds
        grad_params[2*i+1] = gx * dx_dt + gy * dy_dt
        
    return -grad_params # We want to maximize area, so minimize negative area

def _objective_and_grad(params):
    """
    Objective and gradient for L-BFGS-B.
    """
    points = _param_to_points(params)
    val = -_min_triangle_area_vectorized(points)
    grad = _grad_min_area_params(params)
    return val, grad

def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    This creates a well-spread set of points, which is a good starting point
    for maximizing the minimum triangle area.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        # Generate random candidates
        # Uniform sampling in triangle:
        # r1, r2 in [0,1]
        # if r1 + r2 > 1, r1 = 1-r1, r2 = 1-r2
        # point = r1*V1 + r2*V2
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        # Calculate min distance from each candidate to existing points
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Pick candidate with max min_dist
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _lagrange_multipliers_residual(params, active_set):
    """
    Compute the residual for the Lagrange multiplier system.
    We want to maximize min area subject to points in triangle.
    At optimum, for the active constraints (triangles with min area),
    the gradient of the objective w.r.t. each point is zero (if interior)
    or balanced by boundary constraints.
    
    Here we simplify: assume all active triangles are interior to the domain.
    The condition is that the sum of gradients of active area functions,
    weighted by Lagrange multipliers, equals zero for each point coordinate.
    
    However, a simpler and more robust approach for non-convex problems is
    to use the fact that at a local maximum of the minimum of a set of functions,
    there exist weights w_i >= 0 summing to 1 such that the gradient of the
    weighted sum is zero.
    
    Let f_k(x) be the area of triangle k.
    We want to solve: sum_k w_k * grad f_k(x) = 0 for all x_j,
    subject to w_k >= 0, sum w_k = 1.
    
    This is a quadratic program in w for a fixed x.
    But we can also view it as: find x and w such that the KKT conditions hold.
    
    For the residual function in scipy.optimize.root, we need a square system.
    Variables: x (2N) + w (M_active).
    Equations:
    1. grad_x (sum w_k f_k) = 0  (2N equations)
    2. sum w_k = 1 (1 equation)
    3. w_k * (f_k - t) = 0 (complementarity) -- hard to handle directly.
    
    Alternative: Just use the gradient of the smoothed min or the active set gradient.
    If we assume a fixed active set A, then at the optimum:
    grad (sum_{k in A} f_k / |A|) = 0 is NOT correct.
    The correct condition is: There exists w in simplex such that sum w_k grad f_k = 0.
    
    Let's define the residual for a given active set A.
    We solve for x and w.
    Residual components:
    1. For each point j and coordinate d: sum_{k in A} w_k * (d/dx_{jd}) f_k = 0.
    2. sum_{k in A} w_k = 1.
    
    This gives 2N + 1 equations.
    Unknowns: 2N (x) + |A| (w).
    If |A| != 1, the system is not square.
    
    Actually, the number of active constraints at the optimum is typically small.
    For 11 points, the number of triangles is 165.
    The "critical" triangles are those defining the minimum area.
    Usually, 3-5 triangles are active.
    
    Let's try a different approach:
    Use the "smallest area" function.
    The gradient of min(f_1, ..., f_M) at a point where f_1 is uniquely minimum is grad f_1.
    If multiple are minimum, it's the convex hull of their gradients.
    
    A common heuristic is to solve for x such that the minimum area is stationary.
    This can be approximated by finding x where the gradient of the average of the
    smallest m areas is zero, for some small m.
    
    Let's implement a solver that assumes a fixed number of active constraints m,
    and tries to balance their gradients.
    """
    points = _param_to_points(params)
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Identify active set
    min_area = np.min(areas)
    # Tolerance for "active"
    tol = 1e-6
    active_mask = areas <= min_area + tol
    active_indices = np.where(active_mask)[0]
    
    if len(active_indices) == 0:
        return None, None
        
    # If unique minimum, just return gradient of that area
    if len(active_indices) == 1:
        idx = active_indices[0]
        # Gradient of area w.r.t. points
        # This is complex to compute vectorized for a single idx easily in this context
        # We'll rely on Nelder-Mead for the heavy lifting and use Lagrange only as a polish
        # if we can identify a stable active set.
        pass

    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # 1. Generate diverse initializations
    # FPS initializations are generally very good for packing problems
    n_fps_inits = 15
    initial_params = []
    
    for i in range(n_fps_inits):
        if time_left() < 15.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add a few random initializations to ensure diversity
    for i in range(5):
        if time_left() < 15.0:
            break
        params = rng.uniform(0.1, 0.9, size=2*N)
        initial_params.append(params)

    # Also include the current best known heuristic structure if we had one, 
    # but FPS is usually superior. We'll stick to FPS + Random.

    # 2. Local Polish with L-BFGS-B
    # We refine all generated candidates.
    
    for start_p in initial_params:
        if time_left() < 5.0:
            break
            
        try:
            res = minimize(
                _objective_and_grad, start_p, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Final Polish on the best solution found
    if best_params is not None and time_left() > 10.0:
        try:
            res = minimize(
                _objective_and_grad, best_params, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: Center points
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5

    points = _param_to_points(best_params)
    return np.array(points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
