"""Rewritten program: Riesz s-energy + smooth LSE covering radius refinement via L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
RIESZ_S = 12.0
WALL_CLOCK_LIMIT = 145.0  # safety margin below 150s


def _riesz_energy_and_grad(points, s=RIESZ_S):
    """
    Compute the Riesz s-energy and its gradient for N points on the unit sphere.
    Energy: sum_{i != j} |p_i - p_j|^{-s}
    Gradient w.r.t. p_i: -s * sum_{j != i} |p_i - p_j|^{-s-2} * (p_i - p_j)
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    mask = ~np.eye(n, dtype=bool)
    energy = np.sum(dist[mask] ** (-s))

    inv_dist_s = dist ** (-s)
    np.fill_diagonal(inv_dist_s, 0.0)
    inv_dist_s2 = inv_dist_s / dist_sq
    np.fill_diagonal(inv_dist_s2, 0.0)

    grad = -s * np.sum(inv_dist_s2[:, :, np.newaxis] * diff, axis=1)

    return energy, grad


def _smooth_covering_radius_and_grad(points, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    For each point y on a fine grid, compute min_j angle(y, p_j).
    The smooth max is approximated via log-sum-exp over grid points.
    
    We use a coarser grid for speed and rely on the gradient to refine.
    """
    # Generate a moderately fine grid for the smooth approximation
    n_theta = 80
    n_phi = 160
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH)
    ]).reshape(-1, 3)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # YP[i,j] = Y_i . P_j, so angle = arccos(YP)
    YP = Y @ points.T  # shape (n_grid, N)
    # Clip for numerical safety
    YP = np.clip(YP, -1.0, 1.0)
    angles = np.arccos(YP)  # shape (n_grid, N)
    
    # For each grid point, min angle to any config point
    min_angles = np.min(angles, axis=1)  # shape (n_grid,)
    
    # Log-sum-exp approximation of max(min_angles)
    # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_cr = log_sum_exp / beta  # in radians
    
    # Gradient: d/dp_j of smooth_cr
    # d/dp_j [ (1/beta) log sum_i exp(beta * min_k angle(i,k)) ]
    # This is complex due to the min. Approximate by weighting each grid point
    # by its softmax weight, and for each grid point, the gradient w.r.t. p_j
    # is non-zero only for the nearest point (or we can use soft-min).
    
    # Soft-min approach: instead of hard min, use soft-min with parameter gamma
    gamma = 30.0
    # soft_min_j = - (1/gamma) log sum_k exp(-gamma * angle(i,k))
    # But we need the gradient w.r.t. p_j of the max over i of soft_min_i
    
    # Simpler: weight each grid point by softmax over i, and for each i,
    # weight each j by softmax over k of -angle(i,k) (i.e., soft-nearest)
    # grad_cr ≈ (1/beta) * sum_i w_i * sum_j v_ij * grad_{p_j} angle(i,j)
    # where w_i = exp(beta * min_angle_i) / sum exp(beta * min_angle_k)
    # and v_ij = exp(-gamma * angle(i,j)) / sum_k exp(-gamma * angle(i,k))
    
    # Compute weights w_i
    w_i = np.exp(scaled - max_scaled)
    w_i = w_i / np.sum(w_i)
    
    # Compute soft-nearest weights v_ij
    neg_angles = -gamma * angles  # shape (n_grid, N)
    max_neg = np.max(neg_angles, axis=1, keepdims=True)
    exp_neg = np.exp(neg_angles - max_neg)
    v_ij = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # shape (n_grid, N)
    
    # Gradient of angle(i,j) w.r.t. p_j:
    # angle = arccos(Y_i . P_j)
    # d/dP_j arccos(c) = -1/sqrt(1-c^2) * Y_i
    # So grad_{P_j} angle(i,j) = -Y_i / sqrt(1 - (Y_i.P_j)^2)
    
    # We need: grad_cr = (1/beta) * sum_i w_i * sum_j v_ij * (-Y_i / sqrt(1 - YP_ij^2))
    # Note: this is the gradient w.r.t. P_j, but we want it projected to tangent space later.
    
    # For efficiency, compute per-point contributions
    # grad_j = (1/beta) * sum_i w_i * v_ij * (-Y_i / sqrt(1 - YP_ij^2))
    
    # sqrt(1 - YP^2) can be zero at antipodal points; add small epsilon
    denom = np.sqrt(np.maximum(1.0 - YP * YP, 1e-12))  # shape (n_grid, N)
    
    # grad_j = -(1/beta) * sum_i w_i * v_ij * Y_i / denom_ij
    # This is a weighted sum of Y_i vectors
    # weights_ij = w_i * v_ij / denom_ij
    weights_ij = w_i[:, np.newaxis] * v_ij / denom  # shape (n_grid, N)
    
    # grad_j = -(1/beta) * sum_i weights_ij * Y_i
    # = -(1/beta) * (weights_ij.T @ Y)  -- but weights_ij is (n_grid, N), Y is (n_grid, 3)
    # grad[j] = -(1/beta) * sum_i weights_ij[i,j] * Y[i]
    # = -(1/beta) * (weights_ij.T @ Y)[j]
    
    grad = -(1.0 / beta) * (weights_ij.T @ Y)  # shape (N, 3)
    
    return smooth_cr, grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _spherical_lloyd_iteration(points, n_iters=20):
    """
    Perform Lloyd's algorithm on the sphere: for each point, move to the centroid
    of its Voronoi cell (approximated by the average of nearby points weighted by
    inverse distance). This is a simple but effective local refinement.
    """
    points = points.copy()
    for _ in range(n_iters):
        # For each point, compute a weighted average of all other points
        # with weights proportional to 1/dist^2 (Voronoi-like)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Weights: 1/dist^2, but exclude self
        weights = 1.0 / dist_sq
        np.fill_diagonal(weights, 0.0)
        
        # New position: weighted average of all points
        # new_p_i = sum_j weights_ij * p_j / sum_j weights_ij
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.maximum(weight_sums, 1e-15)
        new_points = (weights[:, :, np.newaxis] * points[np.newaxis, :, :]).sum(axis=1) / weight_sums
        points = _project_to_sphere(new_points)
    return points


def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute-force on a fine grid
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2 * np.pi, 400)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH),
            np.sin(TH) * np.sin(PH),
            np.cos(TH)
        ]).reshape(-1, 3)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        YP = Y @ points.T
        min_dist_sq = np.min(2.0 - 2.0 * YP, axis=1)
        max_dist = np.sqrt(np.max(min_dist_sq))
        return np.degrees(np.arccos(np.clip(1.0 - 0.5 * max_dist * max_dist, -1.0, 1.0)))
    
    facets = hull.simplices
    max_ang = 0.0
    for tri in facets:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        M = np.array([
            [a @ a, a @ b, a @ c],
            [b @ a, b @ b, b @ c],
            [c @ a, c @ b, c @ c]
        ])
        rhs = np.array([a @ a, b @ b, c @ c])
        try:
            coeffs = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        O = coeffs[0] * a + coeffs[1] * b + coeffs[2] * c
        O_norm = np.linalg.norm(O)
        if O_norm < 1e-15:
            continue
        O = O / O_norm
        cos_ang = np.clip(a @ O, -1.0, 1.0)
        ang = np.arccos(cos_ang)
        if ang > max_ang:
            max_ang = ang
    return np.degrees(max_ang)


def _run_riesz_optimization(initial_points, deadline):
    """
    Run L-BFGS-B optimization of the Riesz energy on the sphere.
    """
    x0 = initial_points.flatten()
    
    def objective(x):
        points = x.reshape(N, 3)
        points = _project_to_sphere(points)
        energy, grad = _riesz_energy_and_grad(points)
        grad_proj = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        return energy, grad_proj.flatten()
    
    remaining = deadline - time.time()
    max_iter = int(min(5000, max(500, remaining * 100)))
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    x_opt = result.x.reshape(N, 3)
    x_opt = _project_to_sphere(x_opt)
    return x_opt


def _run_lse_refinement(initial_points, deadline, beta=50.0):
    """
    Run L-BFGS-B optimization of the smooth LSE covering radius surrogate.
    """
    x0 = initial_points.flatten()
    
    def objective(x):
        points = x.reshape(N, 3)
        points = _project_to_sphere(points)
        cr, grad = _smooth_covering_radius_and_grad(points, beta=beta)
        grad_proj = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        return cr, grad_proj.flatten()
    
    remaining = deadline - time.time()
    max_iter = int(min(3000, max(300, remaining * 80)))
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    x_opt = result.x.reshape(N, 3)
    x_opt = _project_to_sphere(x_opt)
    return x_opt


def _fibonacci_points(n=N, jitter=0.0, seed=42):
    """Generate Fibonacci spiral points with optional jitter."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        noise = rng.normal(size=(n, 3)) * jitter
        pts = pts + noise
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _random_points(n=N, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + WALL_CLOCK_LIMIT
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(42)
    
    configs = []
    
    # 1. Fibonacci spiral (no jitter)
    configs.append(_fibonacci_points(N, jitter=0.0))
    
    # 2. Jittered Fibonacci (multiple jitter levels)
    for jit in [0.05, 0.1, 0.2]:
        configs.append(_fibonacci_points(N, jitter=jit, seed=42 + int(jit * 100)))
    
    # 3. Random configurations
    for seed in [0, 1, 2, 3]:
        configs.append(_random_points(N, seed=seed))
    
    best_points = None
    best_radius = np.inf
    
    # Phase 1: Riesz energy optimization from multiple starts
    for idx, init in enumerate(configs):
        if time.time() > deadline - 40.0:
            break
        
        try:
            optimized = _run_riesz_optimization(init, deadline)
            radius = _compute_covering_radius(optimized)
            if radius < best_radius:
                best_radius = radius
                best_points = optimized.copy()
        except Exception:
            continue
    
    if best_points is None:
        return _fibonacci_points(N)
    
    # Phase 2: Lloyd iteration refinement
    if time.time() < deadline - 20.0:
        try:
            lloyd_pts = _spherical_lloyd_iteration(best_points, n_iters=15)
            radius_lloyd = _compute_covering_radius(lloyd_pts)
            if radius_lloyd < best_radius:
                best_radius = radius_lloyd
                best_points = lloyd_pts
        except Exception:
            pass
    
    # Phase 3: Smooth LSE covering radius refinement
    if time.time() < deadline - 15.0:
        try:
            lse_pts = _run_lse_refinement(best_points, deadline, beta=50.0)
            radius_lse = _compute_covering_radius(lse_pts)
            if radius_lse < best_radius:
                best_radius = radius_lse
                best_points = lse_pts
        except Exception:
            pass
    
    # Phase 4: Final Riesz polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            polished = _run_riesz_optimization(best_points, deadline)
            radius_polished = _compute_covering_radius(polished)
            if radius_polished < best_radius:
                best_points = polished
        except Exception:
            pass
    
    # Phase 5: Another LSE pass if time permits
    if time.time() < deadline - 3.0:
        try:
            lse_pts2 = _run_lse_refinement(best_points, deadline, beta=100.0)
            radius_lse2 = _compute_covering_radius(lse_pts2)
            if radius_lse2 < best_radius:
                best_points = lse_pts2
        except Exception:
            pass
    
    return best_points
