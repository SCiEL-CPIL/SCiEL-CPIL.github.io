import numpy as np

N = 26

def construct():
    """
    Packs 26 circles in a unit square maximizing the sum of radii.
    Uses a "Grow and Relax" physical simulation approach.
    1. Initialize circles in a jittered grid.
    2. Iteratively increase radii and relax positions to resolve overlaps.
    """
    best_sum = -1
    best_result = None

    # Run multiple trials to escape local minima
    num_trials = 20
    
    for trial in range(num_trials):
        # 1. Initialization: Jittered Grid
        # 6x5 grid = 30 points, take 26
        cols = 6
        rows = 5
        xs = np.linspace(0.1, 0.9, cols)
        ys = np.linspace(0.1, 0.9, rows)
        
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        
        # Shuffle to break symmetry
        indices = np.random.permutation(len(grid_x))
        x = grid_x[indices[:N]].copy()
        y = grid_y[indices[:N]].copy()
        
        # Add small random jitter
        x += np.random.uniform(-0.05, 0.05, N)
        y += np.random.uniform(-0.05, 0.05, N)
        
        # Initial small radius
        r = np.full(N, 0.02)
        
        # Simulation parameters
        num_growth_steps = 500
        growth_rate = 0.0008
        relaxation_iters = 10
        alpha = 0.05 # Step size for relaxation
        
        for step in range(num_growth_steps):
            # Grow radii
            r += growth_rate
            
            # Relax positions to resolve overlaps
            for _ in range(relaxation_iters):
                # Compute forces
                Fx = np.zeros(N)
                Fy = np.zeros(N)
                
                # 1. Pairwise Repulsion
                dx = x[:, np.newaxis] - x[np.newaxis, :]
                dy = y[:, np.newaxis] - y[np.newaxis, :]
                dist_sq = dx**2 + dy**2
                dist = np.sqrt(dist_sq)
                
                # Avoid division by zero
                dist[dist == 0] = 1e-6
                
                # Overlap amount: max(0, r_i + r_j - dist)
                r_sum = r[:, np.newaxis] + r[np.newaxis, :]
                overlap = np.maximum(0, r_sum - dist)
                
                # Force direction: unit vector from j to i
                unit_x = dx / dist
                unit_y = dy / dist
                
                # Zero out diagonal
                np.fill_diagonal(overlap, 0)
                np.fill_diagonal(unit_x, 0)
                np.fill_diagonal(unit_y, 0)
                
                # Sum forces
                Fx += np.sum(overlap * unit_x, axis=1)
                Fy += np.sum(overlap * unit_y, axis=1)
                
                # 2. Boundary Repulsion
                # Push in if too close to wall
                Fx += np.maximum(0, r - x)
                Fx -= np.maximum(0, x - (1 - r))
                
                Fy += np.maximum(0, r - y)
                Fy -= np.maximum(0, y - (1 - r))
                
                # Update positions
                x += alpha * Fx
                y += alpha * Fy
                
                # Clamp positions to [0, 1] to prevent instability
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
            
            # Check validity and update best
            valid = True
            # Boundary check with tolerance
            if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5):
                valid = False
            if np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5):
                valid = False
                
            if valid:
                # Overlap check
                dx = x[:, np.newaxis] - x[np.newaxis, :]
                dy = y[:, np.newaxis] - y[np.newaxis, :]
                dist = np.sqrt(dx**2 + dy**2)
                r_sum = r[:, np.newaxis] + r[np.newaxis, :]
                np.fill_diagonal(r_sum, 0)
                if np.any(dist < r_sum - 1e-5):
                    valid = False
            
            if valid:
                current_sum = np.sum(r)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = np.column_stack((x, y, r))
                    
    if best_result is None:
        # Fallback to seed-like grid if optimization fails completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
