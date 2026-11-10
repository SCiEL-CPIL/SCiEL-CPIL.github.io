# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import itertools


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill

    # Helper: compute score R = (d_min / d_max)^2
    def score(points):
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Objective for minimizer: minimize -score (i.e., maximize score)
    # But since score is non-smooth (min/max), we use a different approach
    # We minimize a penalty based on maximizing d_min while keeping d_max fixed, or use a proxy

    # Better approach: maximize min_dist / max_dist
    # We can reformulate: for a given configuration, we want to maximize min(dists) / max(dists)
    # This is equivalent to maximizing the ratio.
    # A common trick: minimize -log(min_dist) + log(max_dist) = log(max_dist/min_dist)
    # Or simply maximize min_dist subject to max_dist <= 1 (by scaling), then the ratio is min_dist.
    # Since the problem is scale-invariant, we can fix the maximum distance to 1 and maximize the minimum distance.
    # However, fixing max distance to 1 is a constraint.
    # Alternative: Use the objective f(x) = -min_dist(x) + max_dist(x) * epsilon? No.
    
    # Let's use the objective: minimize (max_dist - min_dist) is not quite right.
    # The ratio r = d_min / d_max. We want to maximize r.
    # Let's try to maximize d_min subject to d_max <= 1.
    # If we find the max d_min with d_max=1, that's the best ratio.
    
    # Objective: minimize -d_min(x) + penalty(d_max(x) > 1)
    # Or simpler: minimize (d_max(x) - d_min(x)) / d_max(x) = 1 - d_min/d_max. Minimizing this maximizes the ratio.
    # Let g(x) = 1 - (d_min / d_max). We want to minimize g(x).
    # g(x) is 0 when all distances are equal (equidistant set), which is impossible in 2D for 16 points.
    # g(x) is non-negative.

    def objective(flat):
        pts = flat.reshape(n, d)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 1.0
        return 1.0 - (d_min / d_max)

    # Initial best points
    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Generate diverse initial configurations
    def generate_initial_configs():
        configs = []
        
        # 1. Grid-like
        np.random.seed(0)
        for _ in range(5):
            pts = np.random.randn(n, d)
            configs.append(pts)
        
        # 2. Circle
        for _ in range(5):
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.random.rand()
            radii = np.random.rand(n) * 0.5 + 0.5
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            configs.append(pts)
            
        # 3. Two clusters
        for _ in range(5):
            c1 = np.random.randn(8, d)
            c2 = np.random.randn(8, d) + [10, 10]
            pts = np.vstack([c1, c2])
            configs.append(pts)
            
        # 4. Hexagonal packing approximation
        for _ in range(5):
            pts = np.array([
                [0, 0], [1, 0], [0.5, np.sqrt(3)/2], [1.5, np.sqrt(3)/2],
                [0, np.sqrt(3)], [1, np.sqrt(3)], [0.5, np.sqrt(3)*1.5], [1.5, np.sqrt(3)*1.5],
                [0, np.sqrt(3)*2], [1, np.sqrt(3)*2], [0.5, np.sqrt(3)*2.5], [1.5, np.sqrt(3)*2.5],
                [0, np.sqrt(3)*3], [1, np.sqrt(3)*3], [0.5, np.sqrt(3)*3.5], [1.5, np.sqrt(3)*3.5]
            ])
            pts = pts + np.random.randn(n, d) * 0.1
            configs.append(pts)
            
        # 5. Random with scaling
        for _ in range(5):
            pts = np.random.rand(n, d) * 10
            configs.append(pts)
            
        return configs

    configs = generate_initial_configs()
    
    # Refine each configuration using local optimization
    for i, config in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Use Powell method which doesn't require gradients
        try:
            res = minimize(objective, config.flatten(), method='Powell', 
                           options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
            if res.success:
                update_best(res.x.reshape(n, d))
        except Exception:
            pass
        
        # Also try Nelder-Mead
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, config.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success:
                update_best(res.x.reshape(n, d))
        except Exception:
            pass
            
    # If we have time, do a more global search or more iterations
    # We can also try to perturb the best found and re-optimize
    if best_points is not None and time.time() < deadline - 60:
        np.random.seed(123)
        for _ in range(20):
            if time.time() > deadline - 60:
                break
            perturbed = best_points + np.random.randn(n, d) * 0.5
            try:
                res = minimize(objective, perturbed.flatten(), method='Powell',
                               options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8})
                if res.success:
                    update_best(res.x.reshape(n, d))
            except Exception:
                pass

    if best_points is None:
        # Fallback to random
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points


# EVOLVE-BLOCK-END
