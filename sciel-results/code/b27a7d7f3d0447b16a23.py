"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialization: Greedy Grid Packing
    # Create a dense grid of points inside the unit circle
    # We want to pick N points that are well-spread.
    
    # Generate a fine grid
    grid_size = 100
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Filter points inside the unit circle (with a small margin to be safe initially)
    dists_origin = np.hypot(grid_points[:, 0], grid_points[:, 1])
    grid_points = grid_points[dists_origin <= 1.0]
    
    # Greedy selection: Pick the point that maximizes the minimum distance to already picked points
    # Start with the center? Or a corner? 
    # A common heuristic is to start with the point furthest from the origin (corner) or just center.
    # Let's start with the origin (0,0) as it's symmetric, or better, pick a random point from the grid to break symmetry?
    # Actually, for packing, starting with a point on the boundary or center is fine.
    # Let's try starting with (0,0).
    
    selected_points = []
    # Add center
    selected_points.append([0.0, 0.0])
    available_indices = np.arange(len(grid_points))
    
    # We need to select N-1 more points
    for _ in range(N - 1):
        # Calculate min distance from each available grid point to all selected points
        # selected_points is a list of [x, y]
        sel_arr = np.array(selected_points)
        # dists[i, j] is distance between grid_point i and selected_point j
        diff = grid_points[:, np.newaxis, :] - sel_arr[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Find the grid point with the maximum minimum distance
        best_idx = np.argmax(min_dists)
        
        # Add this point
        selected_points.append(grid_points[best_idx].copy())
        
        # Remove this point from available (optional, for efficiency, but grid is small enough)
        # available_indices = np.delete(available_indices, best_idx)
        
    points = np.array(selected_points)
    
    # 2. Simulated Annealing Optimization
    
    def get_radius(pts):
        # Distance to boundary
        d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        r_boundary = np.min(d_boundary)
        
        # Pairwise distances
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        return min(r_boundary, r_pair)
    
    current_r = get_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Annealing parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    max_iters = 10000
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if T < T_min:
            break
            
        # Pick a random point to move
        idx = rng.integers(0, N)
        
        # Generate a random perturbation
        # Scale perturbation by temperature
        scale = T * 2.0
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        
        old_pos = points[idx].copy()
        new_pos = old_pos + np.array([dx, dy])
        
        # Check if new position is inside the unit circle
        if np.linalg.norm(new_pos) > 1.0:
            continue # Reject move immediately
            
        points[idx] = new_pos
        
        new_r = get_radius(points)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            # Revert move
            points[idx] = old_pos
            
        # Cool down
        T *= cooling_rate
        
    return best_points
# EVOLVE-BLOCK-END
