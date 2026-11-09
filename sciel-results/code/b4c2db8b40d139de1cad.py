# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a multi-start local optimization approach with:
    1. Sphere-constrained optimization (fixes d_max, maximizes d_min)
    2. Smooth soft-min surrogate for gradient computation
    3. Basin hopping with multiple initializations
    4. Final high-precision polish pass
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 15.0  # seconds
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    # Soft-min function: approximates min of values using log-sum-exp
    def soft_min(values, beta=50.0):
        """Approximate minimum using log-sum-exp trick."""
        if len(values) == 0:
            return 0.0
        # Use the fact that -soft_max(-values) = soft_min(values)
        # soft_max(x) = (1/beta) * log(sum(exp(beta * x_i)))
        # For numerical stability, subtract max first
        x = -values  # negate for soft_max
        x_max = np.max(x)
        if not np.isfinite(x_max):
            return 0.0
        exp_x = np.exp(beta * (x - x_max))
        log_sum = np.log(np.sum(exp_x))
        soft_max_val = (x_max + log_sum) / beta
        return -soft_max_val
    
    def compute_dists_flat(x):
        """Compute all pairwise distances from flat vector."""
        points = x.reshape(n, d)
        return pdist(points)
    
    def objective_sphere(x):
        """
        Objective for sphere-constrained optimization.
        We constrain all points to unit sphere, so d_max is bounded.
        Maximize d_min by minimizing negative soft-min of distances.
        """
        points = x.reshape(n, d)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        points = points / norms
        
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
    
    def neg_objective_sphere(x):
        """Negative objective for minimization on sphere."""
        return -objective_sphere(x)
    
    def smooth_neg_objective(x, beta=50.0):
        """
        Smooth surrogate: minimize -soft_min(distances) / max_distance
        This is differentiable and guides optimization toward
        configurations with larger minimum distance.
        """
        points = x.reshape(n, d)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        points = points / norms
        
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        
        # Use soft-min as smooth approximation of min distance
        sm = soft_min(dists, beta)
        d_max = dists.max()
        
        if d_max <= 0 or not np.isfinite(sm) or not np.isfinite(d_max):
            return 0.0
        
        # We want to maximize sm/d_max, so minimize -(sm/d_max)
        return -(sm / d_max)
    
    # Initialization strategies
    init_strategies = []
    
    # Strategy 1: Fibonacci sphere
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
    
    # Strategy 2: Random uniform on sphere
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    # Strategy 3: Icosahedron-based start
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
    
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("symmetric", symmetric_start(n)))
    
    # Add random sphere starts
    for seed in range(15):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    # Add perturbed symmetric starts
    for seed in range(5):
        rng = np.random.default_rng(seed + 1000)
        base = symmetric_start(n)
        perturbation = rng.standard_normal((n, 3)) * 0.1
        perturbed = base + perturbation
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed /= norms
        init_strategies.append((f"perturbed_sym_{seed}", perturbed))
    
    # Optimize each initialization
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        # First, try basin hopping with the smooth objective
        try:
            result = basinhopping(
                smooth_neg_objective,
                x0,
                minimizer_kwargs={
                    'method': 'BFGS',
                    'options': {
                        'maxiter': 500,
                        'gtol': 1e-8
                    }
                },
                niter=30,
                stepsize=0.5,
                T=10.0,
                accept_test=None,
                seed=42
            )
        except Exception:
            result = None
        
        if result is not None and hasattr(result, 'x'):
            optimized_points = result.x.reshape(n, d)
            # Normalize to sphere
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            optimized_points = optimized_points / norms
            
            score = _score(optimized_points)
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
        
        # Then, try direct optimization with the true objective
        if not check_deadline():
            try:
                result2 = basinhopping(
                    neg_objective_sphere,
                    x0,
                    minimizer_kwargs={
                        'method': 'Powell',
                        'options': {
                            'maxiter': 2000,
                            'xatol': 1e-10,
                            'fatol': 1e-12
                        }
                    },
                    niter=20,
                    stepsize=0.3,
                    T=5.0,
                    accept_test=None,
                    seed=123
                )
            except Exception:
                result2 = None
            
            if result2 is not None and hasattr(result2, 'x'):
                optimized_points2 = result2.x.reshape(n, d)
                norms = np.linalg.norm(optimized_points2, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                optimized_points2 = optimized_points2 / norms
                
                score2 = _score(optimized_points2)
                if score2 > best_score:
                    best_score = score2
                    best_points = optimized_points2.copy()
    
    # Final polish: if we have a good result, do a high-precision local optimization
    if best_points is not None and not check_deadline():
        x0 = best_points.flatten()
        try:
            # Use the true objective with tight tolerances
            result_polish = minimize(
                neg_objective_sphere,
                x0,
                method='Powell',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14
                }
            )
            if hasattr(result_polish, 'x'):
                polished_points = result_polish.x.reshape(n, d)
                norms = np.linalg.norm(polished_points, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                polished_points = polished_points / norms
                
                score_polish = _score(polished_points)
                if score_polish > best_score:
                    best_score = score_polish
                    best_points = polished_points.copy()
        except Exception:
            pass
        
        # Also try BFGS on the smooth objective for final polish
        if not check_deadline():
            try:
                result_bfgs = minimize(
                    smooth_neg_objective,
                    x0,
                    method='BFGS',
                    options={
                        'maxiter': 10000,
                        'gtol': 1e-12
                    }
                )
                if hasattr(result_bfgs, 'x'):
                    bfgs_points = result_bfgs.x.reshape(n, d)
                    norms = np.linalg.norm(bfgs_points, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-10, 1.0, norms)
                    bfgs_points = bfgs_points / norms
                    
                    score_bfgs = _score(bfgs_points)
                    if score_bfgs > best_score:
                        best_score = score_bfgs
                        best_points = bfgs_points.copy()
            except Exception:
                pass
    
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
