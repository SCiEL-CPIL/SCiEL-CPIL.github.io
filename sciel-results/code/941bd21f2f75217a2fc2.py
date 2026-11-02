"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score_centres(c):
    """Compute the largest common radius for centres c (N,3) in [-0.5,0.5]^3."""
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.abs(c)
    wall_min = wall.min()
    # Pairwise distances
    # Use broadcasting for N=100 -> 100x100x3 which is fine
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    pair_min = dist.min()
    return min(wall_min, pair_min / 2.0)


def _neg_score(c):
    return -_score_centres(c)


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin

    # --- Initial configuration: 5x5x4 grid ---
    # 5*5*4 = 100 points exactly
    # Place them in a grid within [-0.5, 0.5]^3
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 1.0/nx, 0.5 - 1.0/nx, nx)
    ys = np.linspace(-0.5 + 1.0/ny, 0.5 - 1.0/ny, ny)
    zs = np.linspace(-0.5 + 1.0/nz, 0.5 - 1.0/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # grid shape (100, 3)
    
    best_c = grid.copy()
    best_r = _score_centres(best_c)
    
    # --- Optimization using L-BFGS-B with multiple restarts ---
    # We'll try a few perturbations and optimize each
    
    rng = np.random.default_rng(42)
    
    num_restarts = 5
    max_iter_per_restart = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            c0 = best_c.copy()
        else:
            # Perturb the best solution or the grid
            if restart % 2 == 1:
                c0 = grid + rng.normal(0, 0.02, size=grid.shape)
            else:
                c0 = best_c + rng.normal(0, 0.01, size=best_c.shape)
        
        # Clip to stay inside cube (with small margin)
        c0 = np.clip(c0, -0.49, 0.49)
        
        # Use L-BFGS-B
        try:
            res = minimize(
                _neg_score,
                c0,
                method='L-BFGS-B',
                jac=None,  # numerical gradient
                options={
                    'maxiter': max_iter_per_restart,
                    'maxls': 50,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 50000
                }
            )
            if res.success or res.fun < _neg_score(c0):
                c_new = res.x
                # Ensure validity
                if np.all(np.isfinite(c_new)) and c_new.shape == (N, 3):
                    r_new = _score_centres(c_new)
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c_new.copy()
        except Exception:
            pass
    
    # --- Final polish: more iterations on the best solution ---
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _neg_score,
                best_c,
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'maxls': 100,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxfun': 100000
                }
            )
            if res.fun < _neg_score(best_c):
                c_new = res.x
                if np.all(np.isfinite(c_new)) and c_new.shape == (N, 3):
                    r_new = _score_centres(c_new)
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c_new.copy()
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_c)):
        return grid[:N]
    if best_c.shape != (N, 3):
        return grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
