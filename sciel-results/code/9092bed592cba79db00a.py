"""Improved packing algorithm for 50 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    n = len(centers)
    # Wall constraint: distance to nearest wall
    # Square is [-0.5, 0.5]^2
    wall_dist = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    
    # Pairwise distances
    # Compute all pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)


def _hexagonal_init():
    """Initialize with a hexagonal-like pattern."""
    # For 50 circles, try a hexagonal arrangement
    # Rows with alternating offsets
    centers = []
    # Try different row counts
    best_centers = None
    best_r = 0
    
    for rows in range(6, 10):
        cols = int(np.ceil(N / rows))
        # Hexagonal spacing
        # For hex packing, horizontal spacing = 2r, vertical = r*sqrt(3)
        # We want to fit in [-0.5, 0.5]^2
        # Try to estimate r
        # Width needed: (cols-1)*2r + 2r = cols*2r <= 1 -> r <= 1/(2*cols)
        # Height needed: (rows-1)*r*sqrt(3) + 2r <= 1 -> r <= 1/(2 + (rows-1)*sqrt(3))
        r_est = min(1.0 / (2.0 * cols), 1.0 / (2.0 + (rows - 1) * np.sqrt(3)))
        
        # Create hexagonal grid
        pts = []
        for i in range(rows):
            for j in range(cols):
                if len(pts) >= N:
                    break
                x = -0.5 + r_est + j * 2 * r_est
                y = -0.5 + r_est + i * r_est * np.sqrt(3)
                if i % 2 == 1:
                    x += r_est
                pts.append([x, y])
            if len(pts) >= N:
                break
        pts = np.array(pts[:N])
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_centers = pts
    
    return best_centers if best_centers is not None else _grid_init()


def _grid_init():
    """Grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init():
    """Random initialization inside the square."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _objective(x, centers=None):
    """Objective function to minimize (negative radius)."""
    c = x.reshape(N, 2)
    r = _compute_radius(c)
    return -r


def _bounds():
    """Bounds for optimization."""
    return [(-0.5, 0.5)] * (2 * N)


def _simulated_annealing(centers, time_limit=100.0):
    """Simulated annealing to improve the packing."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T = 0.05
    T_min = 0.0001
    cooling = 0.995
    base_step = 0.02
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        idx = np.random.randint(0, N)
        step = np.random.normal(0, base_step, size=2)
        new = current.copy()
        new[idx] += step
        # Clip to valid region
        new[idx] = np.clip(new[idx], -0.5, 0.5)
        
        new_r = _compute_radius(new)
        delta = new_r - current_r
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current = new
            current_r = new_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_r


def _local_optimization(centers, time_limit=50.0):
    """Use scipy minimize to locally optimize the packing."""
    start_time = time.time()
    x0 = centers.flatten()
    
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Try multiple random restarts of local optimization
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Small perturbation for each restart
        if restart == 0:
            x = x0.copy()
        else:
            x = x0 + np.random.normal(0, 0.01, size=2 * N)
            x = np.clip(x, -0.5, 0.5)
        
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=_bounds(),
                             options={'maxiter': 100, 'ftol': 1e-12})
            if result.success or result.fun < -best_r:
                c = result.x.reshape(N, 2)
                r = _compute_radius(c)
                if r > best_r:
                    best = c
                    best_r = r
        except Exception:
            pass
    
    return best, best_r


def construct():
    """Construct the optimal packing of 50 circles in a unit square."""
    start_time = time.time()
    total_time_limit = 140.0  # Safety margin before 150s kill
    
    # Multiple initializations
    inits = []
    inits.append(_grid_init())
    inits.append(_hexagonal_init())
    for _ in range(3):
        inits.append(_random_init())
    
    best_centers = None
    best_r = 0
    
    # Time allocation: SA gets more time, local opt gets less
    sa_time = 80.0
    local_time = 40.0
    
    for init in inits:
        if time.time() - start_time > total_time_limit:
            break
        
        # Simulated annealing
        sa_centers, sa_r = _simulated_annealing(init, time_limit=sa_time)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers
        
        # Local optimization on the SA result
        if time.time() - start_time < total_time_limit - 5:
            local_centers, local_r = _local_optimization(sa_centers, time_limit=local_time)
            if local_r > best_r:
                best_r = local_r
                best_centers = local_centers
    
    # Final polish if time permits
    if best_centers is not None and time.time() - start_time < total_time_limit - 10:
        final_centers, final_r = _local_optimization(best_centers, time_limit=30.0)
        if final_r > best_r:
            best_r = final_r
            best_centers = final_centers
    
    if best_centers is None:
        best_centers = _grid_init()
    
    return best_centers
