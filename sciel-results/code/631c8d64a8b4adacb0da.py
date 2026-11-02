"""Combined program: Smooth surrogate optimization with exact hull polish."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

# Suppress warnings from scipy
warnings.filterwarnings("ignore")

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _make_grid(n_theta=300, n_phi=600):
    """Create a dense spherical grid for smooth surrogate evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

# Pre-compute a grid for the smooth surrogate. 
# Balance between accuracy and speed. 300x600 is ~180k points.
# Matrix mult (180k x 130) is fast in numpy.
_GRID = _make_grid(200, 400) # Start with a moderate grid for speed in early stages
_GRID_FINE = _make_grid(400, 800) # Finer grid for final polish

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Vectorized soft covering radius and its gradient.
    Uses a two-stage soft-min/soft-max to create a smooth surrogate for the max-min distance.
    """
    # Compute angular distances: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min: for each grid point, compute weighted average of angles
    # w_j = exp(-angle_j / sigma) / sum_k exp(-angle_k / sigma)
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
    
    # Soft-min value: weighted sum of angles
    soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
    
    # Soft-max of soft-min values (to get covering radius approximation)
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    # Gradient computation
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)
    
    # Term 1: sum_i coeff[i,j] * (-g_i / denom[i,j])
    alpha = coeff / denom  # (M, N)
    term1 = -(alpha.T @ grid_points)  # (N, 3)
    
    # Term 2: sum_i coeff[i,j] * dots[i,j] / denom[i,j] * p_j
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
    
    grad = term1 + term2  # (N, 3)
    
    return soft_max_val, grad

def _facet_circumradius_batch(pts, simplices):
    """Vectorized computation of facet circumradii for all simplices at once."""
    n_facets = len(simplices)
    if n_facets == 0:
        return np.array([])
    
    v1 = pts[simplices[:, 0]]
    v2 = pts[simplices[:, 1]]
    v3 = pts[simplices[:, 2]]
    
    e1 = v2 - v1
    e2 = v3 - v1
    normals = np.cross(e1, e2)
    n_norms = np.linalg.norm(normals, axis=1)
    
    valid = n_norms > 1e-15
    angles = np.zeros(n_facets)
    
    if np.any(valid):
        d_vals = np.einsum('ij,ij->i', normals[valid], v1[valid])
        dists = np.abs(d_vals) / n_norms[valid]
        dists = np.clip(dists, -1.0, 1.0)
        angles[valid] = np.arccos(dists)
    
    return angles

def _facet_circumcenters(pts, simplices):
    """Vectorized computation of facet circumcenters (unit vectors)."""
    n_facets = len(simplices)
    if n_facets == 0:
        return np.zeros((0, 3))
    
    v1 = pts[simplices[:, 0]]
    v2 = pts[simplices[:, 1]]
    v3 = pts[simplices[:, 2]]
    
    e1 = v2 - v1
    e2 = v3 - v1
    normals = np.cross(e1, e2)
    n_norms = np.linalg.norm(normals, axis=1)
    
    valid = n_norms > 1e-15
    centers = np.zeros((n_facets, 3))
    
    if np.any(valid):
        d_vals = np.einsum('ij,ij->i', normals[valid], v1[valid])
        centers_valid = (d_vals[:, None] / (n_norms[valid] ** 2)[:, None]) * normals[valid]
        c_norms = np.linalg.norm(centers_valid, axis=1)
        valid2 = c_norms > 1e-15
        centers_valid[valid2] /= c_norms[valid2, None]
        
        valid_indices = np.where(valid)[0]
        centers[valid_indices[valid2]] = centers_valid[valid2]
    
    return centers

def _covering_radius(pts):
    """Compute the exact covering radius using convex hull facets."""
    if len(pts) < 4:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi # Fallback, shouldn't happen for valid sphere points
        
    angles = _facet_circumradius_batch(pts, hull.simplices)
    if len(angles) == 0:
        return 0.0
    return np.degrees(np.max(angles))

def _lloyd_iteration(pts, grid_points):
    """One Lloyd iteration: move each point to the centroid of its Voronoi cell."""
    dots = grid_points @ pts.T
    assignments = dots.argmax(axis=1)
    
    new_pts = pts.copy()
    for i in range(len(pts)):
        mask = assignments == i
        if np.sum(mask) > 0:
            centroid = grid_points[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
    
    return new_pts

def _refine_lloyd(pts, grid_points, start_time, max_time, max_iters=10):
    """Lloyd-based refinement."""
    current = pts.copy()
    best = pts.copy()
    
    # Use a fast grid for Lloyd
    fast_grid = _make_grid(100, 200)
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        new_pts = _lloyd_iteration(current, fast_grid)
        
        # Check for coincident points and nudge apart
        for i in range(N):
            for j in range(i + 1, N):
                dot = np.dot(new_pts[i], new_pts[j])
                if dot > 0.999999:
                    diff = new_pts[i] - new_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    diff /= norm
                    new_pts[i] = new_pts[i] + 0.01 * diff
                    new_pts[j] = new_pts[j] - 0.01 * diff
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    new_pts[j] /= np.linalg.norm(new_pts[j])
        
        # Accept if it improves the exact radius (or just take it, Lloyd usually improves)
        # To save time, we only check exact radius every few steps or at the end
        current = new_pts
        best = new_pts.copy()
        
    return best

def _smooth_optimize(pts, start_time, max_time):
    """
    Use L-BFGS-B to minimize the soft covering radius.
    This is the structural change: moving from discrete gradient steps to a smooth continuous optimization.
    """
    current_pts = pts.copy()
    
    # Flatten points for scipy
    x0 = current_pts.flatten()
    
    def objective_and_grad(x):
        p = x.reshape(N, 3)
        # Normalize points to ensure they stay on the sphere during optimization
        # L-BFGS-B works in Euclidean space, so we must project back
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        
        val, grad = _soft_covering_radius_and_grad(p, _GRID, sigma=0.02)
        
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . p) p
        dots = np.sum(grad * p, axis=1, keepdims=True)
        grad_tangent = grad - dots * p
        
        return val, grad_tangent.flatten()
    
    # Use a moderate sigma for the main optimization
    # L-BFGS-B is robust but can get stuck in local minima of the surrogate.
    # We rely on the initial Lloyd steps to get close to the global minimum.
    
    options = {
        'maxiter': 50, # Limit iterations to ensure we finish in time
        'maxfun': 100,
        'ftol': 1e-8,
        'gtol': 1e-6,
        'disp': False
    }
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options=options)
        if res.success or res.status in [0, 1, 2]:
            current_pts = res.x.reshape(N, 3)
            current_pts /= np.linalg.norm(current_pts, axis=1, keepdims=True)
    except Exception:
        pass
    
    return current_pts

def _polish_step(pts, start_time, max_time):
    """
    Final polish: Use exact hull gradient descent for a few steps.
    This corrects for any errors introduced by the grid-based surrogate.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    step_size = 0.01
    no_improve_count = 0
    
    for it in range(50):
        if time.time() - start_time > max_time - 5.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        simplices = hull.simplices
        angles = _facet_circumradius_batch(best_pts, simplices)
        
        if len(angles) == 0:
            break
        
        max_angle = np.max(angles)
        
        # Soft-max weights over facets
        beta = 50.0
        exp_angles = np.exp(beta * (angles - max_angle))
        weights = exp_angles / np.sum(exp_angles)
        
        # Compute circumcenters for all facets
        centers = _facet_circumcenters(best_pts, simplices)
        
        # Compute gradient
        grad = np.zeros_like(best_pts)
        
        significant = weights > 1e-3
        if not np.any(significant):
            break
        
        sig_indices = np.where(significant)[0]
        sig_weights = weights[sig_indices]
        sig_simplices = simplices[sig_indices]
        sig_centers = centers[sig_indices]
        
        for w, simplex, center in zip(sig_weights, sig_simplices, sig_centers):
            for idx in simplex:
                v = best_pts[idx]
                direction = center - np.dot(v, center) * v
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    grad[idx] += w * direction / dir_norm
        
        if np.linalg.norm(grad) > 1e-15:
            grad_norm = np.linalg.norm(grad)
            grad = grad / grad_norm * step_size
            
            new_pts = best_pts + grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                best_pts = new_pts
                best_radius = new_radius
                no_improve_count = 0
                step_size = min(step_size * 1.1, 0.05)
            else:
                step_size *= 0.7
                no_improve_count += 1
                if no_improve_count > 5 or step_size < 0.001:
                    break
    
    return best_pts, best_radius

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 115.0  # Safety margin before 120s limit
    
    # Phase 1: Initial Fibonacci spiral
    pts = _initial_fib()
    
    # Phase 2: Lloyd refinement to get a good starting point for smooth optimization
    # Use a fast grid for Lloyd
    fast_grid = _make_grid(100, 200)
    
    elapsed = time.time() - start_time
    if elapsed < max_time - 60.0:
        pts = _refine_lloyd(pts, fast_grid, start_time, max_time, max_iters=5)
    
    # Phase 3: Smooth optimization using L-BFGS-B on a soft surrogate
    # This is the core structural change.
    elapsed = time.time() - start_time
    if elapsed < max_time - 20.0:
        # Use a moderate grid for the smooth optimization
        # We can try to increase grid resolution if time permits, but start moderate
        global _GRID
        # Re-assign global grid to a slightly finer one for the optimizer
        # But be careful with memory and time. 200x400 = 80k points.
        # 300x600 = 180k points.
        # Let's stick to 200x400 for speed, then polish with exact hull.
        _GRID = _make_grid(200, 400)
        
        pts = _smooth_optimize(pts, start_time, max_time)
        
        # Normalize
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Run a second pass of smooth optimization with a finer grid if time allows
        elapsed = time.time() - start_time
        if elapsed < max_time - 15.0:
            _GRID = _make_grid(300, 600)
            pts = _smooth_optimize(pts, start_time, max_time)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
    # Phase 4: Final polish with exact hull gradient descent
    elapsed = time.time() - start_time
    if elapsed < max_time - 5.0:
        pts, _ = _polish_step(pts, start_time, max_time)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                pts[j] += np.random.randn(3) * 0.001
                pts[j] /= np.linalg.norm(pts[j])
    
    # Final normalization
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    return pts
# EVOLVE-BLOCK-END
