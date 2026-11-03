"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_triangle_area_fast(points):
    """Fully vectorized: compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    # Compute all pairwise cross products
    # For triangles (i,j,k) with i<j<k, area = 0.5*|cross(Pj-Pi, Pk-Pi)|
    # Use broadcasting: (n,n,2) - (n,1,2) gives (n,n,2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 2)
    # cross[i,j,k] = diff[i,j,0]*diff[i,k,1] - diff[i,j,1]*diff[i,k,0]
    # But we need cross of (Pj-Pi) and (Pk-Pi) = diff[i,j] x diff[i,k]
    # Let's compute it differently:
    # For each i, for each j>i, for each k>j:
    # area = 0.5 * |diff[i,j,0]*diff[i,k,1] - diff[i,j,1]*diff[i,k,0]|
    
    # More efficient: compute all cross products via broadcasting
    # cross[i,j,k] = diff[i,j,0] * diff[i,k,1] - diff[i,j,1] * diff[i,k,0]
    # This is (n,n,n) which is 11^3 = 1331, very manageable
    
    # Actually, let's use a simpler approach:
    # For each pair (i,j), compute cross with all other points
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = points[j] - points[i]  # (2,)
            # Cross of d with (points[k] - points[i]) for all k
            diff_k = points - points[i]  # (n, 2)
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]  # (n,)
            # Exclude i and j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _objective_cartesian(flat):
    """Objective for Nelder-Mead: minimize negative min triangle area.
    flat: (2*N,) array of cartesian coordinates.
    Returns: -min_area + penalty if any point is outside triangle."""
    points = flat.reshape(N, 2)
    
    # Check if all points are inside the equilateral triangle
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Point (x,y) is inside if:
    #   y >= 0
    #   y <= sqrt(3)*x  (left edge)
    #   y <= sqrt(3)*(1-x)  (right edge)
    x = points[:, 0]
    y = points[:, 1]
    
    penalty = 0.0
    eps = 1e-8
    # y >= 0
    penalty += np.sum(np.maximum(0, -y - eps)) * 10
    # y <= sqrt(3)*x
    penalty += np.sum(np.maximum(0, y - np.sqrt(3)*x + eps)) * 10
    # y <= sqrt(3)*(1-x)
    penalty += np.sum(np.maximum(0, y - np.sqrt(3)*(1-x) + eps)) * 10
    
    min_area = _min_triangle_area_fast(points)
    return -min_area + penalty


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # Clip y to [0, sqrt(3)/2]
    y = np.clip(y, 0, np.sqrt(3)/2)
    
    # Clip based on left edge: y <= sqrt(3)*x => x >= y/sqrt(3)
    x_min = y / np.sqrt(3)
    # Clip based on right edge: y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    # Also ensure x in [0,1]
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _generate_grid_init(rng, grid_size=5, jitter_scale=0.05):
    """Generate a diverse initial configuration using structured grid + jitter."""
    # Create a grid of candidate positions in barycentric coords
    # Then convert to cartesian and add jitter
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if i + j < grid_size:
                a = i / (grid_size - 1)
                b = j / (grid_size - 1)
                c = 1 - a - b
                if a >= 0 and b >= 0 and c >= 0:
                    # Convert to cartesian
                    x = b * 1.0 + c * 0.5
                    y = c * (np.sqrt(3) / 2)
                    positions.append([x, y])
    
    positions = np.array(positions)
    
    # Select N points from the grid with some randomness
    if len(positions) <= N:
        # Use all grid points and fill remaining with random
        selected = positions.copy()
        while len(selected) < N:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0, np.sqrt(3)/2 * 0.8)
            # Check if inside triangle
            if y <= np.sqrt(3)*x and y <= np.sqrt(3)*(1-x):
                selected = np.vstack([selected, [[x, y]]])
        points = selected[:N]
    else:
        # Randomly select N points from grid
        indices = rng.choice(len(positions), size=N, replace=False)
        points = positions[indices]
    
    # Add jitter
    points = points + rng.normal(0, jitter_scale, size=(N, 2))
    points = _clip_to_triangle(points)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Multi-start Nelder-Mead optimization
    n_starts = 8
    starts_per_time = 15  # seconds per start
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            # First start: use a symmetric initialization
            # Place points at vertices, edge midpoints, and center
            points_init = np.array([
                [0.05, 0.05],      # near A
                [0.95, 0.05],      # near B
                [0.5, np.sqrt(3)/2 - 0.05],  # near C
                [0.5, 0.05],       # midpoint AB
                [0.25, np.sqrt(3)/4],  # midpoint AC
                [0.75, np.sqrt(3)/4],  # midpoint BC
                [0.5, np.sqrt(3)/6],   # center
                [0.3, 0.3],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.4, 0.4],
            ])
            points_init = _clip_to_triangle(points_init)
        else:
            # Subsequent starts: grid-based with varying jitter
            jitter = 0.03 + 0.02 * (start_idx % 3)
            points_init = _generate_grid_init(rng, grid_size=4, jitter_scale=jitter)
        
        # Flatten for optimization
        flat_init = points_init.flatten()
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _objective_cartesian,
                flat_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            # Clip result to triangle
            points_opt = _clip_to_triangle(result.x.reshape(N, 2))
            
            # Evaluate
            score = _min_triangle_area_fast(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
                
        except Exception:
            pass
    
    # If no valid result, use a fallback
    if best_points is None:
        best_points = _clip_to_triangle(np.array([
            [0.05, 0.05], [0.95, 0.05], [0.5, np.sqrt(3)/2 - 0.05],
            [0.5, 0.05], [0.25, np.sqrt(3)/4], [0.75, np.sqrt(3)/4],
            [0.5, np.sqrt(3)/6], [0.3, 0.3], [0.7, 0.3],
            [0.5, 0.5], [0.4, 0.4]
        ]))
        best_score = _min_triangle_area_fast(best_points)
    
    # Final refinement: run one more Nelder-Mead from the best point
    if time.time() < deadline - 5:
        try:
            flat_best = best_points.flatten()
            result = minimize(
                _objective_cartesian,
                flat_best,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            points_final = _clip_to_triangle(result.x.reshape(N, 2))
            score_final = _min_triangle_area_fast(points_final)
            if score_final > best_score:
                best_points = points_final
                best_score = score_final
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
