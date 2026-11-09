# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _negative_score(points):
    """Negative of score for minimization."""
    return -_score(points)

def _powell_local_search(x0, time_budget, rng):
    """
    Perform local search using Powell's method.
    Powell's is derivative-free and robust to nonsmoothness.
    """
    start_time = time.time()
    current_x = x0.copy()
    
    # We minimize -score
    # Powell's method in scipy uses the 'powell' method
    # We need to be careful with time. 
    # We can run it in chunks or just rely on the maxiter limit.
    # A better approach for time control: run minimize with a custom callback that checks time.
    
    class TimeOut(Exception):
        pass

    def callback(xk):
        if time.time() - start_time > time_budget:
            raise TimeOut()

    try:
        res = minimize(
            _negative_score,
            current_x,
            method='Powell',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-12},
            callback=callback
        )
        best_x = res.x
    except TimeOut:
        best_x = current_x
    except Exception:
        best_x = current_x

    return best_x

def _cma_es_global_search(x0, time_budget, rng, sigma0=0.5):
    """
    Perform global search using CMA-ES.
    """
    try:
        import cma
    except ImportError:
        # Fallback to simple random restarts if cma is not available (though it usually is)
        # For now, just return x0 if cma fails to import, or do a simple random walk
        return x0

    start_time = time.time()
    bounds = None # No explicit bounds, but we can normalize later
    
    # CMA-ES options
    opts = {
        'bounds': None,
        'seed': rng.integers(100000),
        'maxiter': 1000,
        'verbose': -9,
        'tolopt': 1e-8,
        'tolfun': 1e-12,
        'stopiters': 100, # Stop if no improvement for 100 iterations
    }

    # Use a wrapper to handle time
    es = cma.CMA(x0, sigma0, opts)
    best_x = x0.copy()
    best_f = _negative_score(best_x)
    
    while not es.stop():
        if time.time() - start_time > time_budget:
            break
        
        X = es.ask()
        # Evaluate
        fvals = []
        for x in X:
            f = _negative_score(x)
            fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        es.tell(X, fvals)

    return best_x

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2

    # Hard deadline: 600s wall-clock, leave 30s margin
    deadline = time.time() + 570

    best_points = None
    best_score = -1.0

    rng = np.random.default_rng(42)

    initial_configs = []

    # Config 1: Triangular Lattice (High Quality Start)
    rows = [1, 2, 3, 4, 3, 2, 1]
    points = []
    s = 1.0
    dy = s * np.sqrt(3) / 2
    for i, count in enumerate(rows):
        y = -i * dy
        for j in range(count):
            x = (j - (count - 1) / 2.0) * s
            points.append([x, y])
    initial_configs.append(np.array(points))

    # Config 2: Two concentric regular octagons
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r_outer = 1.0
    r_inner = 0.5
    pts1 = np.vstack([
        np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)]),
        np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    ])
    initial_configs.append(pts1)

    # Config 3: Random
    initial_configs.append(rng.standard_normal((n, d)))

    # Config 4: Square grid 4x4 with jitter
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts4 = np.column_stack([gx.ravel(), gy.ravel()])
    pts4 += rng.standard_normal((n, d)) * 0.05
    initial_configs.append(pts4)

    # Config 5: Circle with 16 points
    angles_c = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts5 = np.column_stack([np.cos(angles_c), np.sin(angles_c)])
    initial_configs.append(pts5)

    # Phase 1: Local search on diverse initial configs
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break

        # Normalize to unit scale for numerical stability
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale
        
        x0 = init.flatten()
        
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        # Local search (Powell's)
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_budget = min(remaining - 5, 30)  # Cap per-config time for Powell
        improved_x = _powell_local_search(x0, time_budget, rng)
        improved_pts = improved_x.reshape(n, 2)
        
        # Re-normalize
        improved_pts = improved_pts - improved_pts.mean(axis=0)
        scale = np.max(np.linalg.norm(improved_pts, axis=1))
        if scale > 1e-15:
            improved_pts = improved_pts / scale
            
        improved_score = _score(improved_pts)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # Phase 2: Global search with CMA-ES if time permits
    # CMA-ES is more expensive per iteration but better at global exploration
    # We use the best point found so far as the center, or random restarts
    
    if best_points is not None:
        # Try CMA-ES centered around the best solution found
        x0 = best_points.flatten()
        remaining = deadline - time.time()
        if remaining > 60:
            time_budget = min(remaining - 30, 120)
            improved_x = _cma_es_global_search(x0, time_budget, rng, sigma0=0.1)
            improved_pts = improved_x.reshape(n, 2)
            improved_pts = improved_pts - improved_pts.mean(axis=0)
            scale = np.max(np.linalg.norm(improved_pts, axis=1))
            if scale > 1e-15:
                improved_pts = improved_pts / scale
            improved_score = _score(improved_pts)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_pts.copy()
        
        # Try a few random restarts with CMA-ES
        while time.time() < deadline - 30:
            init = rng.standard_normal((n, d))
            init = init - init.mean(axis=0)
            scale = np.max(np.linalg.norm(init, axis=1))
            if scale > 1e-15:
                init = init / scale
            x0 = init.flatten()
            
            s = _score(init)
            if s > best_score:
                best_score = s
                best_points = init.copy()
            
            remaining = deadline - time.time()
            if remaining < 20:
                break
            time_budget = min(remaining - 10, 30)
            improved_x = _cma_es_global_search(x0, time_budget, rng, sigma0=0.5)
            improved_pts = improved_x.reshape(n, 2)
            improved_pts = improved_pts - improved_pts.mean(axis=0)
            scale = np.max(np.linalg.norm(improved_pts, axis=1))
            if scale > 1e-15:
                improved_pts = improved_pts / scale
            improved_score = _score(improved_pts)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_pts.copy()

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
