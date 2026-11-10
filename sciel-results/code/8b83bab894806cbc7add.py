"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import cos, sin, pi, log
import time

def _score(points):
    """Compute the score: product of ordered-pair distances / 8^8, after scaling to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)  # avoid log(0)
    # Diameter
    diam = np.max(dists)
    if diam < 1e-12:
        return 0.0
    # Scale so diameter = 2
    scale = 2.0 / diam
    # Product of ordered pair distances after scaling
    # log(product) = sum over all ordered pairs (i != j) of log(scale * d_ij)
    # = n*(n-1)*log(scale) + sum_{i!=j} log(d_ij)
    log_prod = n * (n - 1) * log(scale) + np.sum(np.log(dists))
    score = np.exp(log_prod) / (n ** n)
    return score

def _objective(pts_flat):
    """Negative log-product of distances (to minimize), with scaling."""
    n = 8
    pts = pts_flat.reshape(n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
    scale = 2.0 / diam
    log_prod = n * (n - 1) * log(scale) + np.sum(np.log(dists))
    return -log_prod

def construct():
    deadline = time.time() + 120.0  # safety margin before 150s kill
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Start from regular octagon
    regular = np.array([[cos(2 * pi * k / n), sin(2 * pi * k / n)] for k in range(n)])
    best_pts = regular.copy()
    best_score = _score(regular)
    
    # Try multiple random perturbations and optimize
    rng = np.random.default_rng(42)
    
    for attempt in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Create perturbed starting point
        noise_scale = 0.1 + 0.3 * attempt / 15.0
        pts = regular + rng.normal(0, noise_scale, size=(n, 2))
        
        # Optimize
        try:
            from scipy.optimize import minimize
            result = minimize(
                _objective,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_pts = result.x.reshape(n, 2)
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Return as list of [x, y]
    return [[float(p[0]), float(p[1])] for p in best_pts]
# EVOLVE-BLOCK-END
