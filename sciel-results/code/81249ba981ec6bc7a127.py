"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def _compute_radius(positions):
    """Compute the maximum common radius for a given set of positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Compute pairwise distances efficiently
    dists = cdist(positions, positions)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    min_dist = np.min(dists[mask])
    
    # Distance to boundary
    abs_pos = np.abs(positions)
    dist_to_bound = np.min(0.5 - abs_pos, axis=1)
    min_bound = np.min(dist_to_bound)
    
    r = min(min_dist / 2.0, min_bound)
    return r


def _hex_init(n_points=200):
    """Generate a hexagonal lattice initialization, cropped to fit n_points."""
    # Hexagonal packing: rows offset by half spacing
    # We'll generate a large hex grid and pick the best n_points
    # Spacing s determines radius r = s/2 (for interior points)
    # For boundary, r = 0.5 - max(|x|, |y|)
    
    # Generate hex grid with spacing s
    # Try different spacings to get close to 200 points
    best_positions = None
    best_score = -1
    
    for s in np.linspace(0.06, 0.08, 20):
        # Row spacing in hex packing: s * sqrt(3)/2
        row_spacing = s * np.sqrt(3) / 2
        
        # Generate points
        points = []
        # Estimate how many rows we need
        # x range: [-0.5, 0.5], so about 1/s points per row
        # y range: [-0.5, 0.5], so about 1/row_spacing rows
        n_x = int(1.0 / s) + 2
        n_y = int(1.0 / row_spacing) + 2
        
        for j in range(-n_y, n_y + 1):
            y = j * row_spacing
            # Offset every other row
            x_offset = (j % 2) * s / 2
            for i in range(-n_x, n_x + 1):
                x = i * s + x_offset
                if abs(x) <= 0.5 and abs(y) <= 0.5:
                    points.append((x, y))
        
        points = np.array(points)
        
        if len(points) < n_points:
            continue
        
        # If we have more than n_points, select the best n_points
        # Strategy: keep points that maximize minimum distance
        # For now, just take the first n_points (centered)
        # Better: sort by distance from center and take outer ones?
        # Actually, we want to keep points that are well-spread
        # Let's just take a random subset or the central ones
        
        # For simplicity, take the n_points closest to center? No, that's bad.
        # Take random subset?
        np.random.shuffle(points)
        points = points[:n_points]
        
        # Compute score
        r = _compute_radius(points)
        if r > best_score:
            best_score = r
            best_positions = points.copy()
    
    if best_positions is None:
        # Fallback: simple grid
        m = 15
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        dists_to_origin = np.sum(grid**2, axis=1)
        indices_to_drop = np.argsort(dists_to_origin)[:m*m - n_points]
        mask = np.ones(len(grid), dtype=bool)
        mask[indices_to_drop] = False
        best_positions = grid[mask]
    
    return best_positions


def _grid_init(n_points=200):
    """Generate a grid initialization."""
    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) == n_points:
        return grid
    
    if len(grid) > n_points:
        # Drop points closest to center
        dists_to_origin = np.sum(grid**2, axis=1)
        indices_to_drop = np.argsort(dists_to_origin)[:len(grid) - n_points]
        mask = np.ones(len(grid), dtype=bool)
        mask[indices_to_drop] = False
        return grid[mask]
    else:
        # Add random points
        extra = np.random.uniform(-0.4, 0.4, size=(n_points - len(grid), 2))
        return np.vstack([grid, extra])


def _random_init(n_points=200):
    """Generate random initialization."""
    positions = np.random.uniform(-0.45, 0.45, size=(n_points, 2))
    return positions


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate initializations
    inits = []
    
    # 1. Hexagonal initialization
    hex_pos = _hex_init(N)
    inits.append(('hex', hex_pos))
    
    # 2. Random initializations
    for i in range(3):
        rand_init = _random_init(N)
        inits.append((f'rand_{i}', rand_init))
    
    # 3. Grid initialization (fallback)
    grid_init = _grid_init(N)
    inits.append(('grid', grid_init))
    
    best_x = None
    best_r = -1
    
    # Objective function for scipy: minimize -r
    def objective(x):
        x = x.reshape(-1, 2)
        # Check validity
        if np.any(np.abs(x) > 0.5):
            return 1e6
        r = _compute_radius(x)
        return -r
    
    # Optimize from each initialization
    for init_name, x0 in inits:
        if time.time() > deadline:
            break
        
        x_flat = x0.flatten()
        
        try:
            # Use L-BFGS-B with numerical gradient
            res = minimize(
                objective, 
                x_flat, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-4, 'ftol': 1e-8}
            )
            
            x_opt = res.x.reshape(-1, 2)
            
            # Ensure points are inside the square
            x_opt = np.clip(x_opt, -0.5, 0.5)
            
            r_opt = _compute_radius(x_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_x = x_opt
        except Exception as e:
            pass
    
    # If no optimization succeeded, use best initialization
    if best_x is None:
        best_x = inits[0][1]
        best_r = _compute_radius(best_x)
    
    # Final validation: clip to square
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
