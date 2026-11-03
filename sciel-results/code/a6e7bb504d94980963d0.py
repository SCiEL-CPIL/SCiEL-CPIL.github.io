import time
import numpy as np
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def compute_radius(positions):
        """Compute the actual radius from positions."""
        positions = np.asarray(positions)
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        xs = positions[:, 0]
        ys = positions[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        min_boundary_dist = np.min(dist_b)
        
        return min(min_pair_dist / 2, min_boundary_dist)

    def greedy_select(n_points, grid_points, time_budget):
        """
        Greedy best-fit: select n_points from grid_points to maximize
        minimum distance to already-selected points and boundaries.
        """
        if len(grid_points) < n_points:
            # Pad with random points if grid is too small
            extra = np.random.uniform(-0.49, 0.49, size=(n_points - len(grid_points), 2))
            grid_points = np.vstack([grid_points, extra])
        
        # Precompute boundary distances for all grid points
        xs = grid_points[:, 0]
        ys = grid_points[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        boundary_dist = np.minimum(dist_x, dist_y)
        
        selected = []
        remaining = np.arange(len(grid_points))
        
        # First point: pick the one with max boundary distance (center of square)
        best_idx = np.argmax(boundary_dist[remaining])
        first_idx = remaining[best_idx]
        selected.append(first_idx)
        remaining = np.delete(remaining, best_idx)
        
        # For each subsequent point, pick the one that maximizes the minimum distance
        # to all selected points AND to the boundary
        current_min_dist = np.full(len(grid_points), np.inf)
        current_min_dist[first_idx] = 0
        
        for _ in range(n_points - 1):
            if time.time() > time_budget:
                break
            
            # Update current_min_dist: for each remaining point, 
            # min(distance to nearest selected, boundary_dist)
            sel_points = grid_points[selected]
            dists_to_sel = cdist(grid_points[remaining], sel_points)
            min_to_sel = np.min(dists_to_sel, axis=1)
            
            # The effective distance for each remaining point is min(min_to_sel, boundary_dist)
            eff_dist = np.minimum(min_to_sel, boundary_dist[remaining])
            
            # Pick the point with maximum effective distance
            best_local = np.argmax(eff_dist)
            best_global = remaining[best_local]
            selected.append(best_global)
            remaining = np.delete(remaining, best_local)
        
        return grid_points[selected]

    def make_hex_grid(spacing, n_offsets=8):
        """Create hexagonal grid points with various offsets."""
        points_list = []
        # Generate base hex grid
        i_range = range(-15, 16)
        j_range = range(-15, 16)
        
        for offset_i in np.linspace(-spacing/2, spacing/2, n_offsets):
            for offset_j in np.linspace(-spacing/2, spacing/2, n_offsets):
                points = []
                for i in i_range:
                    for j in j_range:
                        x = j * spacing + (i % 2) * spacing / 2 + offset_j
                        y = i * spacing * np.sqrt(3) / 2 + offset_i
                        points.append([x, y])
                points = np.array(points)
                # Filter to square
                mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
                       (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
                points = points[mask]
                if len(points) >= N:
                    points_list.append(points)
        
        return points_list

    def polish_local(positions, max_iter=200):
        """
        Local polish: for each point, try to move it slightly to increase
        the minimum distance to neighbors and boundaries.
        """
        positions = positions.copy()
        n = len(positions)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(n):
                # Compute current min distance for point i
                if i > 0:
                    dists_to_others = np.linalg.norm(positions - positions[i], axis=1)
                    dists_to_others[i] = np.inf
                    min_to_others = np.min(dists_to_others)
                else:
                    min_to_others = np.inf
                
                dist_x = np.minimum(0.5 - positions[i, 0], positions[i, 0] + 0.5)
                dist_y = np.minimum(0.5 - positions[i, 1], positions[i, 1] + 0.5)
                min_to_bound = np.minimum(dist_x, dist_y)
                
                current_min = min(min_to_others, min_to_bound)
                
                # Try small perturbations
                best_pos = positions[i].copy()
                best_min = current_min
                
                # Try moving in 8 directions
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]:
                    for step in [0.001, 0.0005, 0.0001]:
                        new_pos = positions[i].copy()
                        new_pos[0] += dx * step
                        new_pos[1] += dy * step
                        
                        # Check bounds
                        if new_pos[0] < -0.5 or new_pos[0] > 0.5 or \
                           new_pos[1] < -0.5 or new_pos[1] > 0.5:
                            continue
                        
                        # Compute new min distance
                        if i > 0:
                            dists_new = np.linalg.norm(positions - new_pos, axis=1)
                            dists_new[i] = np.inf
                            min_to_others_new = np.min(dists_new)
                        else:
                            min_to_others_new = np.inf
                        
                        dist_x_new = np.minimum(0.5 - new_pos[0], new_pos[0] + 0.5)
                        dist_y_new = np.minimum(0.5 - new_pos[1], new_pos[1] + 0.5)
                        min_to_bound_new = np.minimum(dist_x_new, dist_y_new)
                        
                        new_min = min(min_to_others_new, min_to_bound_new)
                        
                        if new_min > best_min + 1e-12:
                            best_min = new_min
                            best_pos = new_pos.copy()
                
                if not np.array_equal(best_pos, positions[i]):
                    positions[i] = best_pos
                    improved = True
            
            if not improved:
                break
        
        return positions

    # Main algorithm
    best_radius = -np.inf
    best_positions = None
    
    # Generate hex grids with different spacings
    spacings = [0.073, 0.074, 0.075, 0.076, 0.077, 0.078]
    
    for spacing in spacings:
        if time.time() > deadline - 10:
            break
        
        grids = make_hex_grid(spacing, n_offsets=6)
        
        for grid in grids:
            if time.time() > deadline - 8:
                break
            
            # Greedy selection
            positions = greedy_select(N, grid, deadline - 6)
            
            if len(positions) < N:
                continue
            
            # Polish
            positions = polish_local(positions, max_iter=50)
            
            # Compute radius
            r = compute_radius(positions)
            
            if r > best_radius:
                best_radius = r
                best_positions = positions.copy()
    
    # If we haven't found anything good, use a simpler grid
    if best_radius < 0.03:
        # Square grid fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        positions = grid[:N]
        positions = polish_local(positions, max_iter=100)
        r = compute_radius(positions)
        if r > best_radius:
            best_radius = r
            best_positions = positions
    
    # Final polish on best positions
    if best_positions is not None and time.time() < deadline - 3:
        best_positions = polish_local(best_positions, max_iter=100)
        best_radius = compute_radius(best_positions)
    
    # Clamp to ensure inside square
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    else:
        # Last resort: uniform random
        np.random.seed(0)
        best_positions = np.random.uniform(-0.49, 0.49, size=(N, 2))
    
    return best_positions
