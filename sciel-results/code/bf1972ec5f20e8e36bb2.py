"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 120s limit


def _score(centers):
    """Compute the maximum radius for given centers (higher is better)."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds: centers must be inside [-0.5, 0.5]^2
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Minimum distance to boundary
    boundary_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_boundary = np.min(boundary_dists)
    
    # Minimum pairwise distance
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _objective(centers_flat):
    """Negative score for minimization."""
    centers = centers_flat.reshape(N, 2)
    return -_score(centers)


def _make_grid_init(m_rows, m_cols, n_use=N):
    """Create a grid of points in the unit square."""
    xs = (np.arange(m_cols) + 0.5) / m_cols - 0.5
    ys = (np.arange(m_rows) + 0.5) / m_rows - 0.5
    grid = np.array([(x, y) for y in ys for x in xs])
    if len(grid) >= n_use:
        return grid[:n_use].copy()
    # If grid is too small, tile or adjust
    return grid


def _perturb(centers, magnitude=0.01):
    """Add random perturbation."""
    return centers + np.random.randn(*centers.shape) * magnitude


def construct():
    start_time = time.time()
    
    # Initial grid-based configuration
    m = int(np.ceil(np.sqrt(N)))
    centers = _make_grid_init(m, m)
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Try different grid configurations
    grid_configs = [
        (5, 10), (10, 5), (7, 8), (8, 7), (5, 11), (11, 5),
        (6, 9), (9, 6), (4, 13), (13, 4), (5, 10), (10, 5)
    ]
    
    for (rows, cols) in grid_configs:
        if time.time() - start_time > DEADLINE - 30:
            break
        c = _make_grid_init(rows, cols)
        s = _score(c)
        if s > best_score:
            best_score = s
            best_centers = c.copy()
    
    # Local optimization using Powell method (faster than Nelder-Mead for this)
    if time.time() - start_time < DEADLINE - 20:
        flat = best_centers.ravel()
        try:
            res = minimize(
                _objective, flat,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            new_centers = res.x.reshape(N, 2)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Multiple restarts with perturbations
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # Start from best with perturbation
        c = _perturb(best_centers, magnitude=0.005 * (i + 1))
        # Clip to valid range
        c = np.clip(c, -0.49, 0.49)
        
        flat = c.ravel()
        try:
            res = minimize(
                _objective, flat,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            new_centers = res.x.reshape(N, 2)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time remains
    if time.time() - start_time < DEADLINE - 5:
        flat = best_centers.ravel()
        try:
            res = minimize(
                _objective, flat,
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            new_centers = res.x.reshape(N, 2)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Ensure valid return
    result = np.clip(best_centers, -0.5, 0.5)
    return result
