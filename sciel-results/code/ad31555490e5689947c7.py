# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _smooth_metric(points: np.ndarray, alpha=10.0) -> float:
    """
    Smoothed objective to maximize.
    Uses soft-min and soft-max to approximate d_min and d_max smoothly.
    We want to maximize (d_min / d_max)^2.
    Equivalent to minimizing -2 * (log(d_min) - log(d_max)).
    """
    dists = pdist(points)
    # Soft min: approximates min(dists)
    # log-sum-exp trick for numerical stability
    m = np.min(dists)
    soft_min = m + np.log(np.sum(np.exp(dists - m)) / len(dists))
    # Soft max: approximates max(dists)
    M = np.max(dists)
    soft_max = M - np.log(np.sum(np.exp(M - dists)) / len(dists))
    
    if soft_min <= 0 or soft_max <= 0:
        return 0.0
    
    # Return the squared ratio
    return (soft_min / soft_max) ** 2

def _soft_min(values, alpha=10.0):
    """Soft minimum approximation."""
    m = np.min(values)
    return m + np.log(np.sum(np.exp(values - m)) / len(values))

def _soft_max(values, alpha=10.0):
    """Soft maximum approximation."""
    M = np.max(values)
    return M - np.log(np.sum(np.exp(M - values)) / len(values))

def _icosahedron_vertices():
    """Return vertices of a regular icosahedron (12 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600
    
    best_points = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_points, best_score
        try:
            pts = np.asarray(pts, dtype=float)
            if pts.shape != (n, d):
                return
            if not np.all(np.isfinite(pts)):
                return
            score = _evaluate(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass
    
    # Strategy: Start with icosahedron (12 vertices) and add 2 more points
    ico = _icosahedron_vertices()
    
    # Initial best: icosahedron + 2 random points
    np.random.seed(42)
    extra = np.random.randn(2, d)
    initial_points = np.vstack([ico, extra])
    update_best(initial_points)
    
    # Also try: 12 points on icosahedron, then optimize the 2 extra points
    # and all 14 points together using gradient-based optimization
    
    # Phase 1: Optimize only the 2 extra points, keeping icosahedron fixed
    def optimize_extra_points(fixed_pts, max_iters=500):
        """Optimize the last 2 points while keeping the first 12 fixed."""
        current_pts = np.vstack([fixed_pts, np.random.randn(2, d)])
        current_score = _evaluate(current_pts)
        
        step_size = 0.5
        no_improve = 0
        
        for i in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Perturb one of the extra points
            idx = 12 + np.random.randint(2)
            pert = np.random.randn(d) * step_size
            new_pts = current_pts.copy()
            new_pts[idx] += pert
            
            new_score = _evaluate(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                step_size *= 0.95
                if step_size < 1e-6:
                    break
                if no_improve > 100:
                    step_size = 0.3
                    no_improve = 0
        
        update_best(current_pts)
        return current_pts
    
    # Run multiple restarts for extra point optimization
    for r in range(20):
        if time.time() > deadline - 60:
            break
        np.random.seed(100 + r)
        optimize_extra_points(ico)
    
    # Phase 2: Full optimization using gradient descent on smooth objective
    def objective_flat(x_flat, alpha=10.0):
        """Flattened objective function for scipy.optimize.minimize."""
        pts = x_flat.reshape(n, d)
        score = _smooth_metric(pts, alpha)
        return -score  # Negate to minimize
    
    def gradient_flat(x_flat, alpha=10.0):
        """Compute gradient numerically."""
        eps = 1e-7
        grad = np.zeros_like(x_flat)
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_plus[i] += eps
            x_minus = x_flat.copy()
            x_minus[i] -= eps
            grad[i] = (objective_flat(x_plus, alpha) - objective_flat(x_minus, alpha)) / (2 * eps)
        return grad
    
    # Run gradient-based optimization from multiple starting points
    starting_points = [
        np.vstack([ico, np.random.randn(2, d)]),
        np.random.randn(n, d),
        np.vstack([ico[:10], np.random.randn(4, d)]),
    ]
    
    for sp_idx, sp in enumerate(starting_points):
        if time.time() > deadline - 120:
            break
        
        for alpha in [5.0, 10.0, 20.0]:
            if time.time() > deadline - 60:
                break
            
            try:
                x0 = sp.flatten()
                res = minimize(objective_flat, x0, method='BFGS', jac=gradient_flat,
                              args=(alpha,),
                              options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
                x_opt = res.x
                pts_opt = x_opt.reshape(n, d)
                
                # Evaluate and update
                update_best(pts_opt)
                
                # Second pass with sharper alpha
                if time.time() < deadline - 30:
                    res2 = minimize(objective_flat, x_opt, method='BFGS', jac=gradient_flat,
                                   args=(alpha * 2,),
                                   options={'maxiter': 100, 'gtol': 1e-9, 'disp': False})
                    x_opt2 = res2.x
                    pts_opt2 = x_opt2.reshape(n, d)
                    update_best(pts_opt2)
                    
            except Exception:
                pass
    
    # Phase 3: Final local search polish
    if best_points is not None and time.time() < deadline - 10:
        fine_pts = best_points.copy()
        fine_score = _evaluate(fine_pts)
        step_size = 0.01
        np.random.seed(42)
        
        for i in range(2000):
            if time.time() > deadline - 5:
                break
            
            idx = np.random.randint(n)
            pert = np.random.randn(d) * step_size
            new_pts = fine_pts.copy()
            new_pts[idx] += pert
            
            new_score = _evaluate(new_pts)
            
            if new_score > fine_score:
                fine_pts = new_pts
                fine_score = new_score
                step_size = max(step_size * 0.98, 1e-6)
            else:
                step_size = max(step_size * 0.9, 1e-6)
        
        update_best(fine_pts)
    
    # Fallback
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        update_best(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
