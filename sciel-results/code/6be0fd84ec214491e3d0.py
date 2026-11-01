"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if len(centers) < 2:
        return 0.5
    # Distance between all pairs
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_wall_dist = wall_dists.min()
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _negative_radius(centers):
    """Negative of radius for minimization."""
    return -_compute_radius(centers)

def _hex_init(n):
    """Create a hexagonal close-packed initial configuration."""
    centers = []
    # Try to fit hexagonal pattern
    # For n circles, use approximately sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    spacing = 0.95 / (rows + 1)  # slightly less than unit spacing for safety
    row = 0
    col = 0
    while len(centers) < n:
        x = (col + 0.5) * spacing - 0.5
        y = (row + 0.5) * spacing - 0.5
        if row % 2 == 1:
            x += spacing / 2
        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
            centers.append([x, y])
        col += 1
        if col > rows:
            col = 0
            row += 1
    return np.array(centers[:n])

def _square_init(n):
    """Create a square grid initial configuration."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n):
    """Create a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Try multiple initializations and keep the best
    best_centers = None
    best_radius = 0
    
    # Initializations to try
    inits = [
        _hex_init(N),
        _square_init(N),
        _random_init(N),
        _random_init(N),
    ]
    
    for init_centers in inits:
        if time.time() > deadline - 20:
            break
        
        # Multi-stage optimization with different tolerances
        current = init_centers.copy()
        
        # Stage 1: Coarse optimization
        result = minimize(
            _negative_radius,
            current,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        current = np.clip(result.x, -0.5, 0.5)
        r1 = -result.fun
        
        # Stage 2: Medium precision
        if time.time() > deadline - 10:
            break
        result = minimize(
            _negative_radius,
            current,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        current = np.clip(result.x, -0.5, 0.5)
        r2 = -result.fun
        
        # Stage 3: High precision polish
        if time.time() > deadline - 5:
            break
        result = minimize(
            _negative_radius,
            current,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        current = np.clip(result.x, -0.5, 0.5)
        r3 = -result.fun
        
        # Ultra-precise final refinement
        if time.time() > deadline - 2:
            break
        result = minimize(
            _negative_radius,
            current,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-11}
        )
        current = np.clip(result.x, -0.5, 0.5)
        r_final = -result.fun
        
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
