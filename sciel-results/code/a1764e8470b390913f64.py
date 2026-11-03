"""Direct max-min packer for 100 equal spheres in a unit cube."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return min(r_pair, r_margin)

def _neg_direct_radius(x):
    """Direct non-smooth objective: -min(r_pair, r_margin)."""
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e10
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 1e10
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return -min(r_pair, r_margin)

def _grid_init(nx, ny, nz, jitter=0.0):
    """Generate a grid-based initialization with optional jitter."""
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    zs = np.linspace(-0.5 + 0.125, 0.5 - 0.125, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        points += np.random.normal(0, jitter, points.shape)
        points = np.clip(points, -0.5, 0.5)
    return points

def _random_init(scale=0.05):
    """Generate a random initialization within the cube."""
    points = np.random.uniform(-0.5 + scale, 0.5 - scale, (N, 3))
    return points

def _local_slsqp(points, maxiter=200, ftol=1e-8):
    """Locally optimize using SLSQP on the direct non-smooth objective."""
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        result = minimize(_neg_direct_radius, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': ftol})
        if result.fun < _neg_direct_radius(x0):
            return result.x.reshape(N, 3)
        else:
            return points
    except Exception:
        return points

def _coordinate_polish(points, n_iter=200):
    """Coordinate descent polish on the exact objective."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(n_iter):
        improved = False
        for i in range(N):
            for coord in range(3):
                # Try moving in + and - directions
                for direction in [1, -1]:
                    step = 0.01 * (1.0 - it / n_iter)
                    new_point = current[i].copy()
                    new_point[coord] += direction * step
                    new_point = np.clip(new_point, -0.5, 0.5)
                    
                    old_point = current[i].copy()
                    current[i] = new_point
                    new_score = _score(current)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best = current.copy()
                            best_score = new_score
                    else:
                        current[i] = old_point
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Grid-based initializations
    for nx, ny, nz in [(5, 5, 4), (4, 5, 5), (5, 4, 5), (4, 4, 6), (6, 4, 4)]:
        initializations.append(_grid_init(nx, ny, nz, jitter=0.02))
    
    # Random initializations
    for _ in range(5):
        initializations.append(_random_init(scale=0.05))
    
    # Best-first multi-start
    best_points = None
    best_score = 0.0
    candidates = []
    
    for init in initializations:
        candidates.append((init, _score(init)))
    
    # Sort by initial score (best first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for init, init_score in candidates:
        if time.time() > deadline - 20:
            break
        
        # Local optimization with SLSQP
        opt_points = _local_slsqp(init, maxiter=150, ftol=1e-8)
        opt_score = _score(opt_points)
        
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
        
        # If we have a good solution, do coordinate polish
        if best_score > 0.11 and time.time() < deadline - 10:
            polished = _coordinate_polish(best_points, n_iter=100)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_points = polished
                best_score = polished_score
    
    # Final high-precision polish if time remains
    if best_points is not None and time.time() < deadline - 5:
        final_polished = _coordinate_polish(best_points, n_iter=200)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_points = final_polished
            best_score = final_score
    
    if best_points is None:
        # Fallback: return a simple grid
        best_points = _grid_init(5, 5, 4, jitter=0.0)
    
    return best_points

# EVOLVE-BLOCK-END
