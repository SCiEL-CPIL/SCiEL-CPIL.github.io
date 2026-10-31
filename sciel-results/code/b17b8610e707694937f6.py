"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def construct():
    """
    Pack 21 circles in a rectangle with perimeter 4 (w+h=2) to maximize sum of radii.
    
    Approach: Formulate as minimum-weight matching in a bipartite graph where
    circles are matched to high-resolution spatial grid cells. Use Hungarian algorithm
    to find optimal assignment that maximizes the minimum feasible radius.
    Iteratively optimize grid resolution and aspect ratio.
    """
    best_result = None
    best_sum = -1
    
    # Try different aspect ratios (w + h = 2)
    aspect_ratios = np.linspace(0.25, 0.75, 15)  # w/2 from 0.25 to 0.75
    
    # Try different grid resolutions
    grid_resolutions = [8, 12, 16, 20, 24]
    
    for w_frac in aspect_ratios:
        w = 2 * w_frac
        h = 2 - w
        
        # Skip if aspect ratio is too extreme
        if w < 0.3 or h < 0.3:
            continue
        
        for res in grid_resolutions:
            # Create a high-resolution grid of candidate positions
            # We'll use a non-uniform grid that's denser near boundaries
            n_grid_x = res
            n_grid_y = res
            
            # Generate grid points with some margin from boundaries
            margin_x = 0.05 * w
            margin_y = 0.05 * h
            
            xs = np.linspace(margin_x, w - margin_x, n_grid_x)
            ys = np.linspace(margin_y, h - margin_y, n_grid_y)
            
            # Create all grid cell centers
            grid_positions = []
            for xi in xs:
                for yi in ys:
                    grid_positions.append([xi, yi])
            grid_positions = np.array(grid_positions)
            
            n_cells = len(grid_positions)
            
            if n_cells < N:
                continue
            
            # For each circle, compute the "cost" of assigning it to each grid cell
            # Cost is negative of the minimum distance to other assigned circles and boundaries
            # We'll use an iterative approach: start with random assignment, then refine
            
            # Initial assignment: spread circles roughly evenly
            # Sort grid cells and take every n_cells/N-th cell
            indices = np.linspace(0, n_cells - 1, N, dtype=int)
            initial_assignment = indices
            
            # Compute initial radius for this assignment
            assigned_positions = grid_positions[initial_assignment]
            
            # Compute minimum distance between any two assigned circles
            min_dist = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(assigned_positions[i] - assigned_positions[j])
                    if d < min_dist:
                        min_dist = d
            
            # Also check boundary distances
            for i in range(N):
                d_boundary = min(assigned_positions[i][0], w - assigned_positions[i][0],
                                assigned_positions[i][1], h - assigned_positions[i][1])
                if d_boundary < min_dist:
                    min_dist = d_boundary
            
            r = min_dist / 2 - 1e-8
            
            if r <= 0:
                continue
            
            # Now use Hungarian algorithm to improve the assignment
            # We want to maximize the minimum distance, which is equivalent to
            # minimizing the maximum "closeness" cost
            
            # Build cost matrix: cost[i][j] = -distance from circle i's current position to grid cell j
            # But we need to consider pairwise interactions, which is complex for Hungarian.
            
            # Alternative: Use Hungarian to minimize sum of squared distances from a target pattern
            # Target: evenly spaced circles
            
            # Let's try a different approach: use Hungarian to assign circles to cells
            # that maximize the minimum pairwise distance
            
            # For simplicity, let's use a greedy improvement with Hungarian for local optimization
            
            # Build a cost matrix where cost[i][j] represents how "good" it is to put circle i in cell j
            # We'll use the negative of the minimum distance to other circles' current positions
            
            # This is an approximation; let's do multiple iterations
            
            improved_positions = assigned_positions.copy()
            improved_r = r
            
            for iteration in range(5):
                # Build cost matrix for Hungarian algorithm
                # cost[i][j] = -min distance from grid_positions[j] to all other improved_positions
                cost_matrix = np.zeros((N, n_cells))
                
                for i in range(N):
                    for j in range(n_cells):
                        # Distance from grid_positions[j] to all other circles
                        min_d_to_others = np.inf
                        for k in range(N):
                            if k == i:
                                continue
                            d = np.linalg.norm(grid_positions[j] - improved_positions[k])
                            if d < min_d_to_others:
                                min_d_to_others = d
                        
                        # Also consider boundary
                        d_boundary = min(grid_positions[j][0], w - grid_positions[j][0],
                                        grid_positions[j][1], h - grid_positions[j][1])
                        min_d = min(min_d_to_others, d_boundary)
                        
                        # Cost is negative of this distance (we want to maximize min distance)
                        cost_matrix[i][j] = -min_d
                
                # Use Hungarian algorithm to find optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                new_positions = grid_positions[col_ind]
                
                # Compute new minimum distance
                new_min_dist = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(new_positions[i] - new_positions[j])
                        if d < new_min_dist:
                            new_min_dist = d
                
                for i in range(N):
                    d_boundary = min(new_positions[i][0], w - new_positions[i][0],
                                    new_positions[i][1], h - new_positions[i][1])
                    if d_boundary < new_min_dist:
                        new_min_dist = d_boundary
                
                new_r = new_min_dist / 2 - 1e-8
                
                if new_r > improved_r:
                    improved_positions = new_positions
                    improved_r = new_r
                else:
                    break
            
            total_sum = N * improved_r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = improved_positions[:, 0]
                result[:, 1] = improved_positions[:, 1]
                result[:, 2] = improved_r
                best_result = result
    
    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
