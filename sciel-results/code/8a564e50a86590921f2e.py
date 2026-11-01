"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the largest common radius for given centres in [-0.5, 0.5]^2."""
    # Distance between all pairs
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dists = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    min_boundary = np.min(np.min(0.5 - np.abs(pts), axis=1))
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return r


def _hex_init():
    """Initialize with a hexagonal-like packing."""
    # For 50 circles, try a hexagonal arrangement
    # Hexagonal packing in a square
    # Try different numbers of rows
    best_pts = None
    best_r = 0
    
    for rows in range(5, 10):
        cols = int(np.ceil(N / rows))
        # Create hexagonal pattern
        pts = []
        for i in range(rows):
            for j in range(cols):
                if len(pts) >= N:
                    break
                x = (j + 0.5 * (i % 2)) / cols - 0.5
                y = (i + 0.5) / rows - 0.5
                pts.append([x, y])
        pts = np.array(pts[:N])
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts


def _simulated_annealing(pts, time_limit):
    """Simulated annealing to improve the packing."""
    start_time = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    current_pts = pts.copy()
    current_r = best_r
    
    T = 0.01  # Initial temperature
    cooling = 0.999
    step_size = 0.02
    
    n = len(pts)
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Random perturbation
        i = rng.integers(n)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        new_pts = current_pts.copy()
        new_pts[i, 0] += dx
        new_pts[i, 1] += dy
        
        new_r = _compute_radius(new_pts)
        delta = new_r - current_r
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        
        T *= cooling
        if T < 1e-8:
            T = 0.01  # Reheat
    
    return best_pts, best_r


def _local_refine(pts, time_limit):
    """Local refinement using scipy minimize on a smooth surrogate."""
    start_time = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    def objective(x):
        """Minimize negative radius (maximize radius)."""
        p = x.reshape(-1, 2)
        return -_compute_radius(p)
    
    # Use Nelder-Mead for local refinement
    x0 = best_pts.flatten()
    
    while time.time() - start_time < time_limit:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < -best_r:
            best_pts = res.x.reshape(-1, 2)
            best_r = -res.fun
            x0 = res.x
        else:
            break
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    total_time_limit = 115.0  # Leave 5s margin before 120s limit
    
    # Phase 1: Hexagonal initialization
    pts = _hex_init()
    r = _compute_radius(pts)
    
    # Phase 2: Simulated annealing
    sa_time = 60.0
    pts, r = _simulated_annealing(pts, sa_time)
    
    # Phase 3: Local refinement
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 5.0:
        pts, r = _local_refine(pts, remaining - 5.0)
    
    # Final check: ensure points are within bounds
    pts = np.clip(pts, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return pts
# EVOLVE-BLOCK-END
