import numpy as np
import time
from scipy.optimize import minimize

N = 200
DEADLINE = 140.0  # Safety margin for 150s limit

def compute_radius(points):
    """Compute the maximum common radius for given circle centers."""
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def get_soft_min_value_grad(pts, lam):
    """
    Compute value and gradient of the negative soft-min radius.
    We minimize this function to maximize the soft-min radius.
    
    Soft-min of a set of values v_k:
    S(v) = -1/lam * log( sum exp(-lam * v_k) )
    
    We want to minimize F = -S(v) = 1/lam * log( sum exp(-lam * v_k) )
    
    The values v_k are:
    1. d_ij / 2 for all i < j
    2. 4 wall distances per point (x+0.5, 0.5-x, y+0.5, 0.5-y)
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # 1. Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    d_pairs = dist[iu] / 2.0
    
    # 2. Wall distances (4 per point)
    x = pts[:, 0]
    y = pts[:, 1]
    
    w_x_l = x + 0.5
    w_x_r = 0.5 - x
    w_y_b = y + 0.5
    w_y_t = 0.5 - y
    
    # Flatten all constraints
    v = np.concatenate([d_pairs, w_x_l, w_x_r, w_y_b, w_y_t])
    
    # Soft-min value
    v_min = np.min(v)
    shifted = v - v_min
    exp_terms = np.exp(-lam * shifted)
    sum_exp = np.sum(exp_terms)
    
    value = v_min + (1.0 / lam) * np.log(sum_exp)
    
    # Probabilities
    p = exp_terms / sum_exp
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pairwise part
    p_pairs = p[:len(d_pairs)]
    
    p_matrix = np.zeros((n, n))
    p_matrix[iu] = p_pairs
    p_matrix[iu[1], iu[0]] = p_pairs
    
    safe_dist = np.where(dist > 1e-10, dist, 1e-10)
    force_weights = p_matrix / safe_dist
    
    row_sums = np.sum(force_weights, axis=1)
    weighted_pts = force_weights.dot(pts)
    
    grad_pairs = - (pts * row_sums[:, np.newaxis]) + weighted_pts
    
    # Wall part
    p_walls = p[len(d_pairs):]
    
    p_x_l = p_walls[0:n]
    p_x_r = p_walls[n:2*n]
    p_y_b = p_walls[2*n:3*n]
    p_y_t = p_walls[3*n:4*n]
    
    grad_walls_x = -p_x_l + p_x_r
    grad_walls_y = -p_y_b + p_y_t
    
    total_grad = grad_pairs + np.column_stack([grad_walls_x, grad_walls_y])
    
    return value, total_grad

def repulsive_relax(pts, max_iter=200, deadline=None):
    """
    A simple force-directed relaxation to push points apart.
    Uses inverse-square repulsion.
    """
    pts = pts.copy()
    n = pts.shape[0]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1e-10)
        
        # Repulsive force: F = k / d^2 * direction
        # We want to maximize min distance, so repel everything.
        # Force on i from j is proportional to (pts_i - pts_j) / dist^3
        # Let's use 1/dist^2 magnitude, so vector is (pts_i - pts_j) / dist^3
        
        safe_dist_sq = np.where(dist_sq > 1e-10, dist_sq, 1e-10)
        safe_dist = np.sqrt(safe_dist_sq)
        
        # Weight matrix: 1 / dist^3
        w = 1.0 / (safe_dist_sq * safe_dist)
        np.fill_diagonal(w, 0.0)
        
        # Force on i = sum_j w_ij * (pts_i - pts_j)
        # = sum_j w_ij * pts_i - sum_j w_ij * pts_j
        # = pts_i * sum_j w_ij - (W @ pts)_i
        
        row_sums = np.sum(w, axis=1)
        weighted_pts = w.dot(pts)
        
        force_pairs = pts * row_sums[:, np.newaxis] - weighted_pts
        
        # Wall forces: Push away from walls
        # d/dx (x+0.5) = 1. If x is close to -0.5, push right (+x).
        # d/dx (0.5-x) = -1. If x is close to 0.5, push left (-x).
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_x_l = x + 0.5
        d_x_r = 0.5 - x
        d_y_b = y + 0.5
        d_y_t = 0.5 - y
        
        # Force ~ 1/d^2
        # To avoid singularities at wall, clamp distance
        d_x_l_safe = np.maximum(d_x_l, 0.01)
        d_x_r_safe = np.maximum(d_x_r, 0.01)
        d_y_b_safe = np.maximum(d_y_b, 0.01)
        d_y_t_safe = np.maximum(d_y_t, 0.01)
        
        f_x_l = 1.0 / (d_x_l_safe ** 2)
        f_x_r = 1.0 / (d_x_r_safe ** 2)
        f_y_b = 1.0 / (d_y_b_safe ** 2)
        f_y_t = 1.0 / (d_y_t_safe ** 2)
        
        # Force direction
        # Left wall (-0.5): push right (+1)
        # Right wall (0.5): push left (-1)
        # Bottom wall (-0.5): push up (+1)
        # Top wall (0.5): push down (-1)
        
        force_walls_x = f_x_l - f_x_r
        force_walls_y = f_y_b - f_y_t
        
        total_force = force_pairs + np.column_stack([force_walls_x, force_walls_y])
        
        # Adaptive step size
        # Estimate max force magnitude to scale step
        max_f = np.max(np.linalg.norm(total_force, axis=1))
        if max_f < 1e-10:
            break
        step = 0.5 / max_f
        
        pts_new = pts + step * total_force
        pts_new = np.clip(pts_new, -0.5, 0.5)
        
        pts = pts_new
        
    return pts

def hex_init(n_rows):
    """Generate a hexagonal lattice initialization with n_rows rows."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    # Center and scale to fit in [-0.5, 0.5]
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.48 / max_extent
        pts *= scale
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_r = 0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(10):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # 2. Hexagonal
    for n_rows in range(12, 19):
        pts = hex_init(n_rows)
        inits.append(pts)
        
    # 3. Grid
    m = 14
    grid_pts = []
    for i in range(m):
        for j in range(m):
            if len(grid_pts) < N:
                grid_pts.append([i/m - 0.5 + 0.5/m, j/m - 0.5 + 0.5/m])
    if len(grid_pts) >= N:
        grid_pts = np.array(grid_pts[:N])
        grid_pts += np.random.randn(N, 2) * 0.01
        inits.append(grid_pts)
    
    # Optimization loop
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        
        # Pre-relaxation with force-directed method
        pts = repulsive_relax(pts, max_iter=100, deadline=deadline)
        
        # Define objective for L-BFGS-B
        def obj(x):
            x = x.reshape(N, 2)
            val, grad = get_soft_min_value_grad(x, lam=50.0)
            return val, grad.flatten()
        
        # L-BFGS-B
        try:
            res = minimize(obj, pts.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                pts_opt = res.x.reshape(N, 2)
                r = compute_radius(pts_opt)
                if r > best_r:
                    best_r = r
                    best_points = pts_opt.copy()
        except Exception:
            pass
            
    # Final polish with higher lambda if time remains
    if best_points is not None and time.time() < deadline - 10:
        def obj_polish(x):
            x = x.reshape(N, 2)
            val, grad = get_soft_min_value_grad(x, lam=200.0)
            return val, grad.flatten()
            
        try:
            res = minimize(obj_polish, best_points.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.success:
                pts_polished = res.x.reshape(N, 2)
                r_polished = compute_radius(pts_polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_points = pts_polished.copy()
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
