"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centres):
    """Compute the largest common radius for given centres."""
    # Check validity
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 2):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(np.abs(centres) > 0.5 + 1e-12):
        return 0.0
    
    # Distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(centres), axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    # Use efficient computation
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_pair / 2.0, min_boundary)
    return max(r, 0.0)


def _hex_grid_init(n):
    """Generate a hexagonal lattice initial configuration."""
    # Hexagonal packing: rows offset by half spacing
    # For n circles, arrange in roughly square number of rows
    n_rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    
    points = []
    for row in range(n_rows):
        if len(points) >= n:
            break
        # Offset alternating rows
        offset = 0.0 if row % 2 == 0 else 0.5
        n_cols = int(np.ceil(np.sqrt(n / n_rows))) + 1
        for col in range(n_cols):
            if len(points) >= n:
                break
            x = (col + offset) * 1.0 / n_cols - 0.5
            y = row * (np.sqrt(3) / 2) / n_rows - 0.5
            # Scale to fit in unit square
            x = x * 0.98
            y = y * 0.98
            points.append([x, y])
    
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            points.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    
    return np.array(points[:n])


def _random_init(n):
    """Generate random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(n, 2))


def _local_optimize(centres, max_iters=1000, tol=1e-10):
    """Simple local optimization by moving points away from nearest neighbors."""
    centres = centres.copy()
    best_r = _compute_radius(centres)
    
    for it in range(max_iters):
        improved = False
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        for i in range(len(centres)):
            # Find nearest neighbor
            j = np.argmin(dists[i])
            d = dists[i, j]
            
            if d < 1e-12:
                # Coincident points - separate them
                centres[i, 0] += 0.01 * (np.random.random() - 0.5)
                centres[i, 1] += 0.01 * (np.random.random() - 0.5)
                improved = True
                continue
            
            # Direction from j to i
            direction = (centres[i] - centres[j]) / d
            
            # Check boundary
            for k in range(2):
                if centres[i, k] + direction[k] * 0.001 > 0.5:
                    direction[k] = -1.0
                elif centres[i, k] + direction[k] * 0.001 < -0.5:
                    direction[k] = 1.0
            
            # Move slightly in direction
            new_centre = centres[i] + direction * 0.001
            if np.all(np.abs(new_centre) <= 0.5):
                old_centre = centres[i].copy()
                centres[i] = new_centre
                new_r = _compute_radius(centres)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    improved = True
                else:
                    centres[i] = old_centre
        
        if not improved:
            break
    
    return centres, best_r


def construct():
    """Construct optimal packing of 200 equal circles in unit square."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_centres = None
    best_r = 0.0
    
    # Try multiple initializations
    np.random.seed(42)
    
    for init_method in ['hex', 'random']:
        if time.time() > deadline:
            break
        
        if init_method == 'hex':
            centres = _hex_grid_init(N)
        else:
            centres = _random_init(N)
        
        # Local optimization
        if time.time() > deadline:
            break
        
        optimized_centres, r = _local_optimize(centres, max_iters=500)
        
        if r > best_r:
            best_r = r
            best_centres = optimized_centres.copy()
    
    # Additional random restarts if time permits
    while time.time() < deadline - 10.0:
        centres = _random_init(N)
        optimized_centres, r = _local_optimize(centres, max_iters=300)
        
        if r > best_r:
            best_r = r
            best_centres = optimized_centres.copy()
    
    if best_centres is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]
    
    # Final polish: ensure all points are strictly inside
    best_centres = np.clip(best_centres, -0.499999, 0.499999)
    
    return best_centres
# EVOLVE-BLOCK-END
