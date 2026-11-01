import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
DEADLINE = 115.0  # safety margin before 120s limit

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _covering_radius_exact(pts):
    """Compute exact covering radius in degrees using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_ang = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        
        # Circumcenter of spherical triangle
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)
        
        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)
        
        if np.dot(x, outward) < 0:
            x = -x
        
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot_val))
        
        if ang > max_ang:
            max_ang = ang
    
    return max_ang

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _create_grid(n_grid):
    """Create a fixed grid of points on the sphere for soft-max evaluation."""
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

# Pre-compute a grid for the soft-max objective
# 2000 points is a good balance between accuracy and speed
GRID = _create_grid(2000)
BETA = 50.0  # Sharpness of the soft-max

def _soft_covering_radius_and_grad(pts):
    """
    Compute the soft covering radius (log-sum-exp of angular distances)
    and its gradient with respect to the points.
    
    pts: (N, 3) array of points on the unit sphere.
    Returns: (value, grad) where value is in degrees and grad is (N, 3).
    """
    # Dots: (N_grid, N)
    dots = GRID @ pts.T
    # Clip to avoid numerical issues in arccos
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # Angles: (N_grid,)
    angles = np.arccos(dots)
    
    # Soft-max: log-sum-exp of angles
    # To be numerically stable, subtract max
    max_ang = np.max(angles)
    exp_angles = np.exp(BETA * (angles - max_ang))
    sum_exp = np.sum(exp_angles)
    log_sum_exp = np.log(sum_exp) / BETA + max_ang
    
    value = np.degrees(log_sum_exp)
    
    # Gradient
    # d(log_sum_exp)/d(angles_i) = exp(BETA * (angles_i - max_ang)) / sum_exp
    weights = exp_angles / sum_exp  # (N_grid,)
    
    # d(angles_i)/d(pts_j) = -1 / sqrt(1 - dots_i,j^2) * d(dots_i,j)/d(pts_j)
    # d(dots_i,j)/d(pts_j) = GRID_i
    
    # We need to sum over i for each j
    # grad_j = sum_i weights_i * (-1 / sqrt(1 - dots_i,j^2)) * GRID_i
    
    # For efficiency, we can compute this vectorially
    # denom: (N_grid, N)
    denom = np.sqrt(1.0 - dots * dots)
    
    # weights[:, None] * (-1 / denom) * GRID[:, None, :]
    # This is (N_grid, N, 3)
    # Sum over axis 0 to get (N, 3)
    
    # To save memory, we can do it in chunks or just let numpy handle it
    # N_grid=2000, N=50, 3 -> 300,000 floats, which is fine.
    
    grad_term = weights[:, None] * (-1.0 / denom) * GRID[:, None, :]
    grad_cart = np.sum(grad_term, axis=0)  # (N, 3)
    
    # Project gradient to tangent space
    # grad_tangent_j = grad_cart_j - dot(grad_cart_j, pts_j) * pts_j
    dots_grad_pts = np.sum(grad_cart * pts, axis=1, keepdims=True)
    grad_tangent = grad_cart - dots_grad_pts * pts
    
    return value, grad_tangent

def _objective_and_grad(x_flat):
    """
    Objective function for scipy.optimize.minimize.
    x_flat: (N*3,) flattened array of points.
    """
    pts = x_flat.reshape(N, 3)
    # Normalize to ensure points stay on sphere during optimization steps
    # This is a heuristic to handle numerical drift
    pts = np.array([_normalize(p) for p in pts])
    
    val, grad = _soft_covering_radius_and_grad(pts)
    
    # Flatten gradient
    grad_flat = grad.flatten()
    
    return val, grad_flat

def _local_optimize_l_bfgs(pts, time_limit=10.0):
    """Optimize using L-BFGS-B on the tangent space."""
    start_time = time.time()
    x0 = pts.flatten()
    
    # Bounds are not strictly necessary if we normalize, but L-BFGS-B works best with them.
    # However, we are on a manifold. We use the projection trick.
    # We will use 'L-BFGS-B' method.
    
    options = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxls': 20,
        'disp': False,
        # 'maxfun': 10000, # Let it run until time limit or convergence
    }
    
    # We need a wrapper to check time
    class TimeLimitException(Exception):
        pass

    def objective_with_time(x):
        if time.time() - start_time > time_limit:
            raise TimeLimitException()
        return _objective_and_grad(x)
        
    try:
        res = minimize(objective_with_time, x0, method='L-BFGS-B', jac=True, options=options)
        best_pts = res.x.reshape(N, 3)
    except TimeLimitException:
        # If time runs out, use the last known good point? 
        # Scipy's minimize doesn't easily expose the last point on exception.
        # So we should check time inside the loop if we were writing our own.
        # For now, if it times out, we return the initial pts (or a slightly perturbed version).
        # Actually, a better approach is to do manual gradient descent with time checks.
        best_pts = pts
    except Exception as e:
        #print(f"Optimization error: {e}")
        best_pts = pts
    
    # Ensure points are normalized
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    return best_pts

def construct():
    t_start = time.time()
    np.random.seed(42)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_cr = _covering_radius_exact(best_pts)
    
    # Phase 1: Optimize the initial Fibonacci spiral
    remaining = DEADLINE - (time.time() - t_start) - 5
    if remaining > 5:
        opt_pts = _local_optimize_l_bfgs(best_pts, time_limit=min(20.0, remaining))
        cr = _covering_radius_exact(opt_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = opt_pts.copy()
    
    # Phase 2: Multi-start with perturbations
    while time.time() - t_start < DEADLINE - 10:
        # Perturb the best solution
        strength = 0.1 + 0.2 * np.random.random()
        perturbed = best_pts.copy()
        for idx in range(N):
            random_dir = np.random.randn(3)
            tangent = random_dir - np.dot(random_dir, perturbed[idx]) * perturbed[idx]
            tn = np.linalg.norm(tangent)
            if tn > 1e-15:
                tangent = tangent / tn
                perturbed[idx] = _normalize(perturbed[idx] + strength * tangent)
        
        # Local optimize the perturbed version
        remaining = DEADLINE - (time.time() - t_start) - 5
        if remaining < 5:
            break
        time_limit = min(15.0, remaining)
        new_pts = _local_optimize_l_bfgs(perturbed, time_limit=time_limit)
        
        cr = _covering_radius_exact(new_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = new_pts.copy()
    
    # Final check: make sure points are on the unit sphere and valid
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] = _normalize(best_pts[j] + 1e-4 * np.random.randn(3))
    
    return best_pts
