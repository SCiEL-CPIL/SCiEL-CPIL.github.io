"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _pairwise_distances(pts):
    # pts: (N, D)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    # exclude self distances
    d2 = np.where(np.eye(N, dtype=bool), np.inf, d2)
    return d2

def _objective(params):
    # params: flattened (N*D,)
    pts = params.reshape(N, D)
    d2 = _pairwise_distances(pts)
    dmin2 = np.min(d2)
    dmax2 = np.max(d2)
    if dmin2 <= 0:
        return 1e12
    return dmax2 / dmin2

def _make_initial_points(seed):
    rng = np.random.default_rng(seed)
    # Start with a grid, then perturb
    side = int(np.ceil(N ** 0.5))
    grid = np.array(list(__import__('itertools').product(range(side), repeat=D)), dtype=float)
    pts = grid[:N]
    # Center and scale
    pts -= pts.mean(axis=0)
    scale = 1.0 / np.max(np.linalg.norm(pts, axis=1))
    pts *= scale
    # Add small random perturbation
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_score = np.inf
    best_pts = None
    
    # Try multiple initializations with different seeds
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
        
        pts0 = _make_initial_points(seed)
        x0 = pts0.ravel()
        
        # Use Nelder-Mead for derivative-free optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_pts = result.x.reshape(N, D)
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback to grid
        side = int(np.ceil(N ** 0.5))
        grid = np.array(list(__import__('itertools').product(range(side), repeat=D)), dtype=float)
        return grid[:N]
    
    # Final polish with tighter tolerances if time permits
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'maxfev': 50000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_pts = result.x.reshape(N, D)
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
