# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    idx = _precompute_indices(n)
    
    def objective(points):
        """
        Objective function for optimization.
        We want to maximize the Heilbronn score.
        Score = min(|det_k|) / A_hull.
        
        We use a smooth approximation for the minimum:
        min(x) approx -log(sum(exp(-beta * x))) / beta.
        
        So Score_approx = -log(sum(exp(-beta * R_k))) / beta,
        where R_k = |det_k| / A_hull.
        
        We want to maximize Score_approx, so we minimize -Score_approx.
        Cost = log(sum(exp(-beta * R_k))) / beta.
        
        Note: The objective is scale invariant.
        """
        if not np.all(np.isfinite(points)):
            return 1e10
            
        A_hull = _compute_hull_area(points)
        if A_hull < 1e-9:
            return 1e10
            
        dets = _compute_dets(points, idx)
        abs_dets = np.abs(dets)
        
        R = abs_dets / A_hull
        
        beta = 20.0  # Sharpness of the min approximation. Lower is smoother but less accurate.
        
        # Stable log-sum-exp calculation for log(sum(exp(-beta * R)))
        y = -beta * R
        max_y = np.max(y)
        
        # If max_y is very large, exp(y - max_y) is safe.
        # y is negative or zero, so max_y <= 0.
        
        log_sum = np.log(np.sum(np.exp(y - max_y)))
        cost = (max_y + log_sum) / beta
        
        return cost

    def objective_grad(points):
        """
        Compute objective and gradient using finite differences? 
        No, let's use analytic gradient if possible, or rely on L-BFGS-B's numerical gradient?
        L-BFGS-B uses numerical gradient by default if jac is not provided.
        Given the complexity of the derivative of the hull area and the min approximation,
        providing an analytic gradient is error-prone.
        However, numerical gradient for 26 variables is acceptable.
        
        Wait, we can provide a finite difference gradient ourselves to be sure, 
        or just let scipy do it. Let's let scipy do it for simplicity and robustness,
        but we might need to tune eps.
        """
        return objective(points)

    # Initializations
    inits = []
    rng = np.random.default_rng(seed=42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric polygons
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inits.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random points in unit square, scaled
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 4. Grid with jitter
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # 5. Hexagonal lattice snippet
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (np.abs(hex_pts).max() + 1e-9)
    hex_pts = hex_pts[:13]
    inits.append(hex_pts)
    
    # 6. Fibonacci spiral
    angles_fib = np.arange(n) * 2 * np.pi * 0.6180339887
    radii_fib = np.sqrt(np.arange(1, n + 1) / n)
    inits.append(np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)]))
    
    # 7. Random perturbation of regular polygon
    angles_r = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.normal(0, 0.1, n)
    radii_r = np.ones(n) + rng.normal(0, 0.1, n)
    inits.append(np.column_stack([radii_r * np.cos(angles_r), radii_r * np.sin(angles_r)]))

    best_points = None
    best_score = -1.0
    
    # Sort inits by initial score to prioritize better starts
    init_scores = []
    for pts in inits:
        s = compute_score(pts)
        init_scores.append(s)
    order = np.argsort(init_scores)[::-1]
    
    for i in order:
        if time.time() > deadline - 60:  # Leave 60s for final polish
            break
            
        init_pts = inits[i]
        
        # Optimization
        try:
            # Use L-BFGS-B with numerical gradient
            # We optimize the flattened array
            res = minimize(
                objective_grad,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            opt_pts = res.x.reshape((n, 2))
            opt_score = compute_score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
                
        except Exception as e:
            # If optimization fails, try the initial point
            init_score = compute_score(init_pts)
            if init_score > best_score:
                best_score = init_score
                best_points = init_pts.copy()

    # Final Polish with Nelder-Mead on the best found
    if best_points is not None and time.time() < deadline - 10:
        def nm_objective(x):
            pts = x.reshape((n, 2))
            s = compute_score(pts)
            if s == 0.0:
                return 1.0
            return -s
        
        try:
            res_nm = minimize(nm_objective, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            final_pts = res_nm.x.reshape((n, 2))
            final_score = compute_score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
