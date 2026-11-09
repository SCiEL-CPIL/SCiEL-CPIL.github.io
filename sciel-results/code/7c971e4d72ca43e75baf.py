import numpy as np
import time
import math

N = 100
CUBE_HALF = 0.5
DEADLINE_S = 115.0

def _grid_score(points):
    """
    Compute the largest radius r for the given points.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if len(points) < 2:
        return 0.5
    # Pairwise distances
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    # Boundary distances
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    
    return min(min_pair / 2.0, min_boundary)

def _initial_grid():
    """
    Create a 5x4x5 grid.
    """
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _fcc_lattice():
    """
    Generate FCC lattice points within the cube and select 100 best.
    FCC lattice in a cube of side 1:
    Points at (i, j, k) * a + offsets
    Offsets: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) scaled by a
    a = 1 / n where n is number of unit cells per side.
    """
    # Let's try a = 1/4, so 4x4x4 unit cells.
    # Total points in 4x4x4 FCC: 4 * (4*4*4) = 256 points (roughly, need to clip)
    # Actually, let's just generate a dense set of candidate points from FCC-like structures
    # and pick the best 100.
    
    # Method: Generate points on a fine grid, but perturb them towards FCC positions?
    # Or simply generate FCC points.
    
    # FCC basis vectors:
    # v1 = (1, 1, 0) / sqrt(2)
    # v2 = (1, 0, 1) / sqrt(2)
    # v3 = (0, 1, 1) / sqrt(2)
    # This is a rotated coordinate system.
    # It's easier to use the Cartesian FCC:
    # Lattice constant a.
    # Points: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) * a
    # Plus translations by (a, 0, 0), (0, a, 0), (0, 0, a).
    
    # We want to fit these in [-0.5, 0.5].
    # Let's try a few lattice constants.
    
    best_points = None
    best_score = -1
    
    for a in [0.2, 0.25, 0.3, 0.333]:
        # Generate points
        points_list = []
        # Number of cells along each axis
        n_cells = int(1.0 / a) + 1
        # Iterate over cells
        for i in range(-n_cells, n_cells + 1):
            for j in range(-n_cells, n_cells + 1):
                for k in range(-n_cells, n_cells + 1):
                    # 4 basis points
                    for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                        x = i * a + dx * a
                        y = j * a + dy * a
                        z = k * a + dz * a
                        # Center the lattice?
                        # The FCC lattice is naturally centered if we start from 0.
                        # But we need it in [-0.5, 0.5].
                        # Let's shift so that the lattice is roughly centered.
                        # The extent is roughly n_cells * a.
                        # Let's just collect all and filter.
                        points_list.append([x, y, z])
        
        points = np.array(points_list)
        
        # Filter points inside the cube
        mask = (np.abs(points) <= CUBE_HALF).all(axis=1)
        points = points[mask]
        
        if len(points) < N:
            continue
            
        # Calculate score for each point? No, we need a subset of 100.
        # Greedy selection:
        # 1. Sort by distance to boundary? Or just random?
        # Let's try to select points that maximize the minimum distance.
        # This is a hard problem.
        # Heuristic: Select points with largest "local density" or just random?
        # Let's try: Select points that are far from boundaries first?
        # Or simply: Take all points, and run a greedy farthest-point sampling?
        
        # Greedy Farthest Point Sampling (FPS)
        # Start with a random point (or center).
        # Then pick the point farthest from the current set.
        
        # To make it deterministic and fast:
        # 1. Calculate min boundary dist for all points.
        # 2. Pick the one with max min_boundary?
        # 3. Then iteratively pick the point that maximizes min distance to already selected points.
        
        # This is O(N * M) where M is number of candidates.
        # M can be large (hundreds). N=100. So 100 * 500 = 50k ops. Fast.
        
        selected = []
        remaining = points.copy()
        
        # First point: center of cube? Or point closest to center?
        # Let's pick the point closest to (0,0,0).
        dists_to_origin = np.sum(remaining**2, axis=1)
        idx = np.argmin(dists_to_origin)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
        
        for _ in range(N - 1):
            if len(remaining) == 0:
                break
            # Calculate min distance to selected set for each remaining point
            # selected: (k, 3), remaining: (m, 3)
            # d[i, j] = ||remaining[i] - selected[j]||
            d = np.sqrt(np.sum((remaining[:, None, :] - selected[None, :, :]) ** 2, axis=-1))
            min_d = np.min(d, axis=1)
            
            # Also consider boundary distance?
            # The score is min(min_pair/2, min_boundary).
            # If we just maximize min_pair, we might pick points near boundary.
            # So let's maximize min(min_d/2, boundary_dist).
            
            margins = CUBE_HALF - np.abs(remaining)
            min_boundary = np.min(margins, axis=1)
            
            scores = np.minimum(min_d / 2.0, min_boundary)
            
            idx = np.argmax(scores)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        selected = np.array(selected)
        score = _grid_score(selected)
        
        if score > best_score:
            best_score = score
            best_points = selected
            
    if best_points is not None:
        return best_points
    else:
        return _initial_grid()

def construct():
    start_time = time.time()
    
    # Try FCC initialization
    points = _fcc_lattice()
    
    # Add small random noise to break symmetry
    rng = np.random.default_rng(42)
    points = points + rng.normal(0, 0.005, points.shape)
    points = np.clip(points, -CUBE_HALF, CUBE_HALF)
    
    best_points = points.copy()
    best_score = _grid_score(points)
    
    # Optimization parameters
    step_size = 0.01
    step_decay = 0.999
    min_step = 1e-6
    
    # Main loop
    while time.time() - start_time < DEADLINE_S:
        # Calculate current min distance
        d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        
        # Calculate wall distances
        margins = CUBE_HALF - np.abs(points)
        min_boundary = np.min(margins)
        
        current_score = min(min_pair / 2.0, min_boundary)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # If step size is too small, break
        if step_size < min_step:
            break
            
        # Determine push radius
        # Push points that are close to each other or to the wall
        # We want to increase the minimum distance.
        # So we should push points that are part of the "bottleneck" (min_pair or min_boundary).
        
        # Let's use a potential based approach.
        # Force on i from j: (r_i - r_j) / d_ij^2 if d_ij < threshold.
        # Threshold: 1.5 * min_pair?
        
        push_radius = max(1.5 * min_pair, 1.5 * min_boundary, 0.1)
        
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        mask = (d < push_radius) & (d > 1e-9)
        
        diff = points[:, None, :] - points[None, :, :]
        d_cubed = d**3 + 1e-12
        np.fill_diagonal(d_cubed, 1.0)
        force_vec = diff / d_cubed[:, :, None]
        
        force_vec[~mask] = 0
        
        forces += np.sum(force_vec, axis=1)
        
        # Wall repulsion
        for dim in range(3):
            dist_neg = points[:, dim] + CUBE_HALF
            dist_pos = CUBE_HALF - points[:, dim]
            
            mask_neg = dist_neg < push_radius
            mask_pos = dist_pos < push_radius
            
            f_neg = np.zeros(N)
            f_neg[mask_neg] = 1.0 / (dist_neg[mask_neg]**2 + 1e-12)
            
            f_pos = np.zeros(N)
            f_pos[mask_pos] = 1.0 / (dist_pos[mask_pos]**2 + 1e-12)
            
            forces[:, dim] += f_neg - f_pos
            
        # Normalize forces?
        # The magnitude of forces can be large.
        # Let's scale by step_size.
        points += step_size * forces
        
        # Clip to cube
        points = np.clip(points, -CUBE_HALF, CUBE_HALF)
        
        # Decay step size
        step_size *= step_decay
        
    return best_points
