"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def construct():
    best_result = None
    best_sum = -1
    
    # Set random seed for reproducibility in local search
    np.random.seed(42)
    
    # Try different aspect ratios
    # w + h = 2
    aspect_ratios = np.linspace(0.3, 0.7, 9)
    
    for w_frac in aspect_ratios:
        w = 2 * w_frac
        h = 2 - w
        
        # Create a high-resolution grid for initialization
        # Resolution 40x40 = 1600 cells
        res = 40
        xs = np.linspace(w / (2 * res), w - w / (2 * res), res)
        ys = np.linspace(h / (2 * res), h - h / (2 * res), res)
        
        # Generate all grid points
        grid_points = np.array([[x, y] for x in xs for y in ys])
        M = len(grid_points)
        
        # Calculate distance to boundary for each grid point
        dist_b = np.min(np.stack([
            grid_points[:, 0],
            w - grid_points[:, 0],
            grid_points[:, 1],
            h - grid_points[:, 1]
        ], axis=1), axis=1)
        
        # Use Hungarian algorithm to initialize positions
        # We want to select N points that are well-placed.
        # Cost = -dist_b + noise to encourage diversity in restarts
        # We run multiple restarts with different noise
        
        for restart in range(5):
            # Add random noise to cost to get different initial configurations
            noise = np.random.randn(N, M) * 0.5
            cost_matrix = -dist_b.reshape(1, -1).repeat(N, axis=0) + noise
            
            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Get selected grid points as initial positions
            positions = grid_points[col_ind].copy()
            
            # Local Search to optimize positions
            # Goal: Maximize the minimum feasible radius r
            # r = min( min_i dist_to_boundary_i, min_{i<j} dist_ij ) / 2
            
            # We use a simple hill-climbing / random perturbation method
            best_local_r = -1
            best_local_pos = positions.copy()
            
            for step in range(200):
                # Calculate current radius
                b_dists = np.min(np.stack([
                    positions[:, 0], w - positions[:, 0],
                    positions[:, 1], h - positions[:, 1]
                ], axis=1), axis=1)
                
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                p_dists = np.min(dists, axis=1)
                
                r = np.min(np.minimum(b_dists, p_dists)) / 2.0
                
                if r > best_local_r:
                    best_local_r = r
                    best_local_pos = positions.copy()
                
                # Pick a random circle to move
                i = np.random.randint(N)
                
                # Determine move direction
                # If constrained by boundary, move towards center
                # If constrained by neighbor, move away from nearest neighbor
                if b_dists[i] < p_dists[i]:
                    # Move towards center
                    dx = w / 2 - positions[i, 0]
                    dy = h / 2 - positions[i, 1]
                else:
                    # Move away from nearest neighbor
                    nn = np.argmin(dists[i])
                    dx = positions[i, 0] - positions[nn, 0]
                    dy = positions[i, 1] - positions[nn, 1]
                
                norm = np.sqrt(dx*dx + dy*dy)
                if norm < 1e-6:
                    continue
                
                # Adaptive step size
                step_size = 0.01
                new_pos = positions[i].copy()
                new_pos[0] += (dx/norm) * step_size
                new_pos[1] += (dy/norm) * step_size
                
                # Clip to boundaries
                new_pos[0] = np.clip(new_pos[0], 0, w)
                new_pos[1] = np.clip(new_pos[1], 0, h)
                
                # Apply move
                positions[i] = new_pos
            
            # Update global best
            if best_local_r > best_sum / N:
                best_sum = best_local_r * N
                result = np.zeros((N, 3))
                result[:, 0] = best_local_pos[:, 0]
                result[:, 1] = best_local_pos[:, 1]
                result[:, 2] = best_local_r
                best_result = result

    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
