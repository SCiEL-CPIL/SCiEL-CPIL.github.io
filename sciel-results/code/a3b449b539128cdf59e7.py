"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time


def _score(pts):
    """Compute the maximum radius for a given set of points."""
    if pts.shape[0] != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Check if all points are inside the square [-0.5, 0.5]^2
    if np.any(np.abs(pts) > 0.5):
        return -1.0
    # Minimum distance between any pair of circles
    d = squareform(pdist(pts))
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d)
    # Minimum distance to the boundary
    min_to_boundary = np.min(0.5 - np.abs(pts))
    r = 0.5 * min(min_dist, min_to_boundary)
    return r


def _construct_grid():
    """Create a simple grid layout as a starting point."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _construct_hexagonal():
    """Create a hexagonal packing layout as a starting point."""
    # For 50 circles, we can try a hexagonal arrangement
    # Let's create a hex grid and take the first N points
    points = []
    # Hexagonal grid with spacing 1
    dx = 1.0
    dy = np.sqrt(3) / 2
    
    # We need to fit 50 circles in a unit square
    # Let's try different densities
    for n_rows in range(4, 12):
        for offset in [0, 0.5]:
            count = 0
            pts = []
            for i in range(n_rows):
                for j in range(n_rows + 1):
                    x = (j + (0.5 if i % 2 else 0)) * dx - (n_rows * dx) / 2
                    y = i * dy - (n_rows * dy) / 2
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        pts.append([x, y])
                        count += 1
                        if count >= N:
                            break
                if count >= N:
                    break
            if count >= N:
                pts = np.array(pts[:N])
                # Scale to fit in the square
                # Find the bounding box
                min_coords = np.min(pts, axis=0)
                max_coords = np.max(pts, axis=0)
                width = max_coords[0] - min_coords[0]
                height = max_coords[1] - min_coords[1]
                scale = min(1.0 / width, 1.0 / height)
                pts = (pts - (min_coords + max_coords) / 2) * scale
                return pts
    return _construct_grid()


def _optimize_from_initial(pts0, max_iter=2000):
    """Optimize the packing starting from pts0."""
    def objective(x):
        pts = x.reshape(N, 2)
        # We want to maximize the minimum radius
        # This is equivalent to minimizing the negative radius
        # But we need to handle the non-smooth min
        # Use a smooth approximation: -log-sum-exp of the constraints
        d = squareform(pdist(pts))
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        min_to_boundary = np.min(0.5 - np.abs(pts))
        r = 0.5 * min(min_dist, min_to_boundary)
        return -r
    
    # Use Nelder-Mead for derivative-free optimization
    result = minimize(objective, pts0.flatten(), method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x.reshape(N, 2)


def _local_polish(pts, max_iter=5000):
    """Perform a local polish using a more precise method."""
    def objective(x):
        pts = x.reshape(N, 2)
        d = squareform(pdist(pts))
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        min_to_boundary = np.min(0.5 - np.abs(pts))
        r = 0.5 * min(min_dist, min_to_boundary)
        return -r
    
    # Use Powell's method for a more thorough local search
    result = minimize(objective, pts.flatten(), method='Powell',
                      options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-14})
    return result.x.reshape(N, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Start with multiple initial configurations
    initial_configs = [
        _construct_grid(),
        _construct_hexagonal(),
    ]
    
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(3):
        pts = _construct_grid() + np.random.randn(N, 2) * 0.01
        initial_configs.append(pts)
    
    best_pts = initial_configs[0]
    best_score = _score(best_pts)
    
    # Optimize from each initial configuration
    for i, pts0 in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Coarse optimization
        pts = _optimize_from_initial(pts0, max_iter=1000)
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts
        
        # Local polish if time permits
        if time.time() > deadline - 20:
            continue
        
        pts_polished = _local_polish(pts, max_iter=3000)
        score_polished = _score(pts_polished)
        
        if score_polished > best_score:
            best_score = score_polished
            best_pts = pts_polished
    
    # Final check and return
    final_score = _score(best_pts)
    if final_score < 0:
        # Fallback to grid if something went wrong
        best_pts = _construct_grid()
    
    return best_pts
# EVOLVE-BLOCK-END
