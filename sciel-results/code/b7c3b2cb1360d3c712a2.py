# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    n = 16
    d = 2
    
    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params, p=40.0):
        """
        Smooth surrogate using power-mean approximation.
        Approximates -R = -(d_min/d_max)^2 using smooth power means.
        """
        if time.time() > start_time + deadline - safety_margin:
            return 1e10
            
        points = params.reshape(n, d)
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 1e10
        
        # Avoid zero distances
        dists = np.maximum(dists, 1e-12)
        m = len(dists)
        
        # Power mean approximation
        d_max_approx = (np.sum(dists ** p) / m) ** (1.0 / p)
        d_min_approx = (np.sum(dists ** (-p)) / m) ** (-1.0 / p)
        
        if d_max_approx < 1e-12:
            return 1e10
            
        R_approx = (d_min_approx / d_max_approx) ** 2
        return -R_approx

    def exact_objective(params):
        """Exact objective for final evaluation."""
        if time.time() > start_time + deadline - safety_margin:
            return 1e10
        points = params.reshape(n, d)
        return -compute_R(points)

    def local_minimizer_smooth(params):
        """Local minimizer using L-BFGS-B with smooth objective."""
        res = minimize(
            smooth_objective, 
            params, 
            method='L-BFGS-B', 
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x, res.fun, res.nfev

    def local_minimizer_exact(params):
        """Local minimizer using Powell with exact objective."""
        res = minimize(
            exact_objective, 
            params, 
            method='Powell', 
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        return res.x, res.fun, res.nfev

    best_points = None
    best_score = -np.inf
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    def make_initial_points(seed):
        rng_local = np.random.RandomState(seed)
        config_type = seed % 5
        
        if config_type == 0:
            # 4x4 grid
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.ravel(), Y.ravel()))
        elif config_type == 1:
            # Regular hexagonal packing approximation
            points = np.zeros((n, d))
            for i in range(n):
                row = i // 4
                col = i % 4
                points[i, 0] = col * 1.0 + (row % 2) * 0.5
                points[i, 1] = row * 0.866
            points -= points.mean(axis=0)
        elif config_type == 2:
            # Two concentric circles
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
            points = np.vstack([
                np.column_stack((0.6 * np.cos(angles1), 0.6 * np.sin(angles1))),
                np.column_stack((1.6 * np.cos(angles2), 1.6 * np.sin(angles2)))
            ])
        elif config_type == 3:
            # Fibonacci spiral
            golden_angle = np.pi * (3 - np.sqrt(5))
            angles = np.arange(n) * golden_angle
            radii = np.sqrt(np.arange(n) / n) * 1.5
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            points -= points.mean(axis=0)
        else:
            # Random
            points = rng_local.uniform(-2, 2, size=(n, d))
        
        return points

    # Phase 1: Basin Hopping with Smooth Objective
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > start_time + deadline - safety_margin:
            break
        
        init_points = make_initial_points(seed=100 + restart)
        init_params = init_points.flatten()
        
        try:
            bh = basinhopping(
                smooth_objective, 
                init_params, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-10}},
                niter=40,  # Number of jumps
                stepsize=0.4,
                T=1.0,
                seed=100 + restart
            )
            
            # Evaluate the best point found by basin hopping
            candidate_points = bh.x.reshape(n, d)
            score = compute_R(candidate_points)
            
            if score > best_score:
                best_score = score
                best_points = candidate_points.copy()
                
        except Exception:
            pass

    # Phase 2: Local Refinement with Exact Objective on Best Candidates
    # We also run a few direct Powell refinements from the initial points to catch cases where BH fails
    if time.time() < start_time + deadline - safety_margin:
        for restart in range(n_restarts):
            init_points = make_initial_points(seed=100 + restart)
            init_params = init_points.flatten()
            
            # Powell on exact objective
            try:
                res = minimize(
                    exact_objective, 
                    init_params, 
                    method='Powell', 
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                candidate_points = res.x.reshape(n, d)
                score = compute_R(candidate_points)
                if score > best_score:
                    best_score = score
                    best_points = candidate_points.copy()
            except Exception:
                pass

    # Phase 3: Final High-Precision Polish
    if best_points is not None:
        if time.time() < start_time + deadline - safety_margin:
            try:
                # Use Powell for final polish as it handles the non-smooth min/max well
                res = minimize(
                    exact_objective, 
                    best_points.flatten(), 
                    method='Powell', 
                    options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14}
                )
                candidate_points = res.x.reshape(n, d)
                score = compute_R(candidate_points)
                if score > best_score:
                    best_score = score
                    best_points = candidate_points.copy()
            except Exception:
                pass

    # Fallback
    if best_points is None:
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        best_points = np.column_stack((X.ravel(), Y.ravel()))

    if best_points.shape != (n, d):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        best_points = np.column_stack((X.ravel(), Y.ravel()))

    return best_points

# EVOLVE-BLOCK-END
