import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # We will use Simulated Annealing to maximize the minimum radius.
    # This is a different approach from the simple hill-climbing in the seed.
    
    best_global_r = 0.0
    best_global_pos = None
    
    n_restarts = 20
    n_iterations = 2000
    
    # Initial temperature
    T_start = 0.01
    T_end = 0.0001
    # Cooling rate
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    for _ in range(n_restarts):
        # Initialization: Hexagonal grid is a good starting point
        pos = generate_hex_grid(N)
        
        # Add small noise to break symmetry
        pos += np.random.normal(0, 0.001, pos.shape)
        pos = np.clip(pos, -0.5, 0.5)
        
        current_r = calculate_radius(pos)
        best_local_r = current_r
        best_local_pos = pos.copy()
        
        T = T_start
        step_size = 0.05
        
        for it in range(n_iterations):
            # Pick a random point
            idx = np.random.randint(N)
            old_pos = pos[idx].copy()
            
            # Generate new position
            step = np.random.normal(0, step_size, 2)
            new_pos = old_pos + step
            
            # Keep inside square
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            pos[idx] = new_pos
            new_r = calculate_radius(pos)
            
            # Simulated Annealing acceptance criterion
            delta_r = new_r - current_r
            if delta_r > 0:
                # Always accept improvements
                current_r = new_r
            else:
                # Accept worse moves with probability exp(delta_r / T)
                if np.random.rand() < np.exp(delta_r / T):
                    current_r = new_r
                else:
                    # Revert
                    pos[idx] = old_pos
            
            if current_r > best_local_r:
                best_local_r = current_r
                best_local_pos = pos.copy()
            
            # Cool down
            T *= alpha
            # Slightly decay step size to refine
            step_size = max(0.001, step_size * 0.9995)
        
        if best_local_r > best_global_r:
            best_global_r = best_local_r
            best_global_pos = best_local_pos
    
    return best_global_pos

def generate_hex_grid(N):
    # Generate a hexagonal grid of points centered in the square
    # We want to pick N points from a dense hex grid
    s = 0.15 # spacing
    points = []
    # Iterate over a range that covers the square
    # Hex grid:
    # x = i*s + (j%2)*s/2
    # y = j*s*sqrt(3)/2
    # Center at 0,0
    
    # Determine bounds
    max_j = int(1.0 / (s * np.sqrt(3) / 2)) + 2
    max_i = int(1.0 / s) + 2
    
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < N:
        # Fallback to random
        return np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Randomly select N points
    indices = np.random.choice(len(points), N, replace=False)
    return points[indices]

def calculate_radius(pos):
    # Calculate min pairwise distance
    dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Calculate min boundary distance
    max_abs = np.max(np.abs(pos))
    boundary_dist = 0.5 - max_abs
    
    return min(min_pairwise / 2.0, boundary_dist)
