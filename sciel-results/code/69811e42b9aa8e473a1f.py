"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 145.0  # safety margin before 150s kill


def _radius_score(centers):
    """Compute the largest common radius for given centers."""
    # Distance between all pairs
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    # Square is [-0.5, 0.5]^2
    dist_to_boundary = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    return min(min_pair, min_boundary)


def _objective(centers_flat):
    """Objective to minimize (negative of radius score)."""
    centers = centers_flat.reshape(-1, 2)
    r = _radius_score(centers)
    return -r


def _hex_init(n_circles):
    """Hexagonal packing initialization."""
    centers = []
    # Hexagonal arrangement in a square
    # Try to fit as many as possible in hexagonal pattern
    # For 50 circles, try different row arrangements
    
    # Method: staggered rows
    # Let's try to find a good hexagonal layout
    # Number of rows
    for n_rows in range(4, 12):
        n_cols = int(np.ceil(n_circles / n_rows))
        # Distribute circles across rows
        # Hexagonal: offset odd rows
        x_spacing = 1.0 / (n_cols + 0.5)
        y_spacing = x_spacing * np.sqrt(3) / 2
        
        # Check if it fits
        total_height = (n_rows - 1) * y_spacing
        if total_height > 0.9:
            continue
        
        centers = []
        for row in range(n_rows):
            # Number of circles in this row
            if row % 2 == 0:
                n_in_row = min(n_cols, n_circles - len(centers))
                offset = 0
            else:
                n_in_row = min(n_cols, n_circles - len(centers))
                offset = x_spacing / 2
            
            for col in range(n_in_row):
                x = -0.5 + x_spacing / 2 + col * x_spacing + offset
                y = -0.5 + y_spacing / 2 + row * y_spacing
                if x < 0.49 and y < 0.49:
                    centers.append([x, y])
        
        if len(centers) >= n_circles:
            return np.array(centers[:n_circles])
    
    # Fallback: simple grid
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n_circles]


def _grid_init(n_circles, jitter=0.0):
    """Grid initialization with optional jitter."""
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    result = grid[:n_circles]
    if jitter > 0:
        result = result + np.random.randn(*result.shape) * jitter
        # Clip to valid range
        result = np.clip(result, -0.49, 0.49)
    return result


def _random_init(n_circles):
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, size=(n_circles, 2))


def _optimize_from(centers, deadline):
    """Run local optimization from given initial centers."""
    n = len(centers)
    x0 = centers.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    best_x = x0.copy()
    best_val = _radius_score(centers)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        if result.success or result.fun < -best_val:
            new_centers = result.x.reshape(-1, 2)
            new_val = _radius_score(new_centers)
            if new_val > best_val:
                best_x = result.x.copy()
                best_val = new_val
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_x.reshape(-1, 2), best_val
    
    # Second pass with tighter tolerance if we have time
    try:
        result2 = minimize(
            _objective,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
        )
        if result2.fun < -best_val:
            new_centers = result2.x.reshape(-1, 2)
            new_val = _radius_score(new_centers)
            if new_val > best_val:
                best_x = result2.x.copy()
                best_val = new_val
    except Exception:
        pass
    
    return best_x.reshape(-1, 2), best_val


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_centers = None
    best_val = -np.inf
    
    # Multiple initializations
    inits = []
    
    # Hexagonal
    hex_c = _hex_init(N)
    inits.append(hex_c)
    
    # Grid
    grid_c = _grid_init(N)
    inits.append(grid_c)
    
    # Jittered grid
    for _ in range(3):
        j_c = _grid_init(N, jitter=0.02)
        inits.append(j_c)
    
    # Random
    for _ in range(5):
        r_c = _random_init(N)
        inits.append(r_c)
    
    # Optimize from each initialization
    for init_c in inits:
        if time.time() > deadline:
            break
        
        centers, val = _optimize_from(init_c, deadline)
        if val > best_val:
            best_val = val
            best_centers = centers
        
        # Check if we've reached a good value
        if best_val > 0.071:
            # Keep optimizing but with less time per restart
            pass
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 5:
        r_c = _random_init(N)
        centers, val = _optimize_from(r_c, deadline)
        if val > best_val:
            best_val = val
            best_centers = centers
    
    if best_centers is None:
        # Fallback to grid
        best_centers = _grid_init(N)
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
