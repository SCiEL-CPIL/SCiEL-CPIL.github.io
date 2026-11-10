"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score(centers):
    """Compute the score: min pair distance / 2, min wall clearance."""
    centers = np.asarray(centers).reshape(-1, 3)
    if centers.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(centers)):
        return -1.0
    # Check inside cube
    if np.any(np.abs(centers) > CUBE_HALF + 1e-9):
        return -1.0
    # Wall clearance
    wall_clear = np.min(CUBE_HALF - np.abs(centers))
    # Pair distances
    # Use broadcasting for N=50 (2500 pairs, manageable)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, wall_clear)

def _objective(x):
    """Objective to minimize: negative of the score (we want to maximize score)."""
    centers = x.reshape(N, 3)
    return -_score(centers)

def _hpc_initial():
    """Generate an initial placement using a hexagonal close-packed inspired layout."""
    # For 50 spheres, try a 5x5x4 arrangement with some jitter
    # Or use a body-centered cubic inspired pattern
    
    # Simple approach: start with a slightly perturbed grid
    # 5x5x4 = 100 spots, pick 50
    m_x, m_y, m_z = 5, 5, 4
    xs = (np.arange(m_x) + 0.5) / m_x - 0.5
    ys = (np.arange(m_y) + 0.5) / m_y - 0.5
    zs = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    # Create grid points
    grid = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid.append([x, y, z])
    grid = np.array(grid)
    
    # We need 50 out of 100. Pick every other one or use a pattern
    # Let's pick a subset that's more spread out
    # Actually, let's just use a 4x4x4 = 64 grid and pick 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # 64 points, pick 50
    # Add small random jitter
    np.random.seed(42)
    jitter = np.random.normal(0, 0.02, size=(64, 3))
    grid = grid + jitter
    # Clip to cube
    grid = np.clip(grid, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return grid[:50]

def _optimize_from_start(start_centers, time_limit):
    """Run local optimization from a given starting point."""
    x0 = start_centers.flatten()
    
    # Use L-BFGS-B with bounds
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    try:
        result = minimize(
            _objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return start_centers, -_score(start_centers)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = -1.0
    
    # Strategy: multiple restarts with different initializations
    # Phase 1: Generate diverse starting points
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            start = _hpc_initial()
        elif restart % 3 == 1:
            # Random uniform
            start = np.random.uniform(-CUBE_HALF + 0.02, CUBE_HALF - 0.02, size=(N, 3))
        elif restart % 3 == 2:
            # Perturbed grid
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            np.random.seed(restart * 12345)
            jitter = np.random.normal(0, 0.03 + restart * 0.001, size=(64, 3))
            start = grid + jitter
            start = np.clip(start, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
            start = start[:50]
        else:
            # Another pattern
            start = np.random.uniform(-CUBE_HALF + 0.02, CUBE_HALF - 0.02, size=(N, 3))
        
        # Ensure valid starting point
        start = np.clip(start, -CUBE_HALF + 0.005, CUBE_HALF - 0.005)
        
        # Run optimization
        opt_centers, opt_score_neg = _optimize_from_start(start, deadline)
        opt_score = -opt_score_neg
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers.copy()
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final polish on the best solution
    if best_centers is not None and time.time() < deadline - 5.0:
        x0 = best_centers.flatten()
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        try:
            result = minimize(
                _objective, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            polish_score = -result.fun
            if polish_score > best_score:
                best_centers = result.x.reshape(N, 3)
                best_score = polish_score
        except Exception:
            pass
    
    # Ensure final validity
    if best_centers is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Final clip to ensure inside cube
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    return best_centers
# EVOLVE-BLOCK-END
