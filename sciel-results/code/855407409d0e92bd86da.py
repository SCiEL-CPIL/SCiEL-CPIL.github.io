# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _objective(pts_flat):
    """Objective function for minimization: (max/min)^2."""
    pts = pts_flat.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return 1e10
    return (dmax / dmin) ** 2

def make_hexagonal_cluster(n, spread=5):
    """
    Generate a hexagonal lattice cluster of n points centered at origin.
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)
    
    # Center the cluster
    pts -= pts.mean(axis=0)
    
    # Sort by distance to center and pick the closest n
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_pts = None
    best_score = np.inf
    
    # 1. Generate initial candidate
    hex_pts = make_hexagonal_cluster(N)
    
    # Evaluate initial score
    s = _score(hex_pts)
    if s < best_score:
        best_score = s
        best_pts = hex_pts.copy()
        
    current_pts = hex_pts.copy()
    current_score = s
    
    # 2. Basin Hopping
    # We alternate between local optimization and random perturbation.
    # This allows escaping local minima.
    
    np.random.seed(42)
    
    while time.time() < deadline - 5:
        # Local Optimization
        try:
            result = minimize(
                _objective,
                current_pts.ravel(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 5000, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < current_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    
                # Update current state to the locally optimized point
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
            
        # Perturbation
        # Add random noise to jump to a new basin
        # The scale of noise should be relative to the current configuration size
        # or just a fixed small value since we are dealing with normalized-like configs
        noise_scale = 0.5 
        current_pts = current_pts + np.random.randn(N, 2) * noise_scale
        
        # Optional: Re-center to avoid drift
        current_pts -= current_pts.mean(axis=0)

    # 3. Final polish on the best found
    if best_pts is not None and time.time() < deadline - 1:
        try:
            result = minimize(
                _objective,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        return make_hexagonal_cluster(N)
        
    return best_pts
# EVOLVE-BLOCK-END
