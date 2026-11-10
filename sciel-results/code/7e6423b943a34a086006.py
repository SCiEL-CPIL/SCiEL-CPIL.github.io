# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a multi-phase optimization:
    1. Powell's method for global search on Cartesian coordinates
    2. L-BFGS-B polish with soft-min objective for precision
    3. Multiple diverse initial configurations
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0  # seconds
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def symmetric_start(n_points):
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    def soft_min_objective(flat_pts, beta=50.0):
        """
        Smooth approximation of -d_min using log-sum-exp.
        Minimizing this maximizes the minimum pairwise distance.
        """
        pts = flat_pts.reshape(-1, 3)
        dists = pdist(pts)
        if len(dists) == 0 or np.any(dists < 1e-15):
            return 1e10
        # Soft min: -log(sum(exp(-beta*d))) / beta approximates -min(d)
        # We want to maximize min(d), so minimize -soft_min(d)
        # Using: -log(sum(exp(-beta*d)))/beta
        # For numerical stability: -log(sum(exp(-beta*(d - d_max))))/beta - d_max
        d_max_val = np.max(dists)
        shifted = -beta * (dists - d_max_val)
        # Clip for numerical stability
        shifted = np.clip(shifted, -700, 700)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min_val = -log_sum / beta + d_max_val
        return -soft_min_val  # Negate so we minimize
    
    def soft_min_gradient(flat_pts, beta=50.0):
        """Numerical gradient of soft_min_objective."""
        eps = 1e-6
        grad = np.zeros_like(flat_pts)
        f0 = soft_min_objective(flat_pts, beta)
        for i in range(len(flat_pts)):
            fp = flat_pts.copy()
            fp[i] += eps
            f_plus = soft_min_objective(fp, beta)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Generate initial configurations
    init_configs = []
    
    # Fibonacci sphere
    init_configs.append(fibonacci_sphere(n))
    
    # Symmetric (icosahedron + 2)
    init_configs.append(symmetric_start(n))
    
    # Random sphere starts with various seeds
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= norms
        init_configs.append(pts)
    
    # Also try slightly perturbed versions of good configs
    for base_idx in range(min(3, len(init_configs))):
        rng = np.random.default_rng(1000 + base_idx)
        perturbed = init_configs[base_idx].copy()
        perturbed += rng.standard_normal(perturbed.shape) * 0.1
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed /= norms
        init_configs.append(perturbed)
    
    # Optimization loop
    for cfg_idx, init_pts in enumerate(init_configs):
        if check_deadline():
            break
        
        flat_init = init_pts.flatten()
        
        # Phase 1: Powell's method (derivative-free, good for global search)
        try:
            result_powell = minimize(
                soft_min_objective,
                flat_init,
                method='Powell',
                args=(80.0,),
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'xtol': 1e-12,
                    'disp': False
                }
            )
            flat_opt = result_powell.x
        except Exception:
            flat_opt = flat_init
        
        if check_deadline():
            break
        
        # Phase 2: L-BFGS-B polish (gradient-based, precise)
        try:
            result_lbgfs = minimize(
                soft_min_objective,
                flat_opt,
                method='L-BFGS-B',
                jac=soft_min_gradient,
                args=(100.0,),
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            flat_opt = result_lbgfs.x
        except Exception:
            pass
        
        if check_deadline():
            break
        
        # Phase 3: Another Powell pass from L-BFGS-B result (escape shallow minima)
        try:
            result_powell2 = minimize(
                soft_min_objective,
                flat_opt,
                method='Powell',
                args=(120.0,),
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'xtol': 1e-14,
                    'disp': False
                }
            )
            flat_opt = result_powell2.x
        except Exception:
            pass
        
        if check_deadline():
            break
        
        # Phase 4: Final L-BFGS-B polish
        try:
            result_lbgfs2 = minimize(
                soft_min_objective,
                flat_opt,
                method='L-BFGS-B',
                jac=soft_min_gradient,
                args=(150.0,),
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'disp': False
                }
            )
            flat_opt = result_lbgfs2.x
        except Exception:
            pass
        
        # Convert back to points and normalize
        opt_pts = flat_opt.reshape(n, 3)
        norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        opt_pts /= norms
        
        score = objective(opt_pts)
        
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
    
    # If no valid result, use Fibonacci sphere as fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
