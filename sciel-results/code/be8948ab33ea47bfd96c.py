"""Improved circle packing for n=32 in unit square."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Initial configuration: hexagonal packing
    # For 32 circles, use a hexagonal grid pattern
    # Optimal packing for small n often has hexagonal symmetry
    
    # Try multiple initial configurations and pick the best
    def hexagonal_init(n_circles):
        """Create hexagonal packing initial positions."""
        positions = []
        # Use a hexagonal grid
        # For n=32, a good arrangement is roughly 6x6 grid with some removed
        # or a more optimized hex pattern
        
        # Start with a simple hexagonal grid
        rows = 6
        cols = 6
        spacing = 1.0 / (cols + 0.5)  # approximate spacing
        
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n_circles:
                    break
                x = (j + 0.5 + (i % 2) * 0.5) * spacing
                y = (i + 0.5) * spacing
                # Keep within bounds
                x = min(max(x, 0.05), 0.95)
                y = min(max(y, 0.05), 0.95)
                positions.append((x, y))
        
        # If we don't have enough, add more
        while len(positions) < n_circles:
            positions.append((0.5, 0.5))
        
        return np.array(positions[:n_circles])
    
    def random_init(n_circles, seed=42):
        """Random initial positions."""
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(n_circles, 2))
    
    def grid_init(n_circles):
        """Grid initial positions."""
        side = int(np.ceil(np.sqrt(n_circles)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= n_circles:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append((x, y))
        while len(positions) < n_circles:
            positions.append((0.5, 0.5))
        return np.array(positions[:n_circles])
    
    def compute_radii(positions):
        """Compute maximum radii for given positions."""
        n = len(positions)
        radii = np.zeros(n)
        
        for i in range(n):
            # Distance to walls
            dist_wall = min(positions[i][0], 1 - positions[i][0],
                           positions[i][1], 1 - positions[i][1])
            max_r = dist_wall
            
            # Distance to other circles
            for j in range(n):
                if i == j:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                max_r = min(max_r, dist / 2.0)
            
            radii[i] = max_r
        
        return radii
    
    def total_radius(positions):
        """Compute sum of radii."""
        return np.sum(compute_radii(positions))
    
    def optimize_positions(positions, max_time=30.0):
        """Optimize positions to maximize sum of radii."""
        start_time = time.time()
        n = len(positions)
        
        # Use gradient-free optimization with local search
        # Simulated annealing style
        current_pos = positions.copy()
        current_val = total_radius(current_pos)
        best_pos = current_pos.copy()
        best_val = current_val
        
        temperature = 0.01
        cooling_rate = 0.995
        step_size = 0.01
        
        while time.time() - start_time < max_time and temperature > 1e-6:
            # Propose a move
            move_idx = np.random.randint(n)
            move_dim = np.random.randint(2)
            delta = np.random.normal(0, step_size)
            
            new_pos = current_pos.copy()
            new_pos[move_idx, move_dim] += delta
            
            # Keep within bounds
            new_pos[move_idx, move_dim] = np.clip(new_pos[move_idx, move_dim], 0.01, 0.99)
            
            new_val = total_radius(new_pos)
            
            if new_val > current_val or np.random.random() < np.exp((new_val - current_val) / temperature):
                current_pos = new_pos
                current_val = new_val
                
                if current_val > best_val:
                    best_pos = current_pos.copy()
                    best_val = current_val
            
            temperature *= cooling_rate
            step_size *= 0.999
        
        return best_pos, best_val
    
    # Try different initializations
    initializations = [
        hexagonal_init(N),
        grid_init(N),
        random_init(N, seed=1),
        random_init(N, seed=42),
        random_init(N, seed=123),
    ]
    
    best_positions = None
    best_value = -np.inf
    
    for init_pos in initializations:
        if time.time() > deadline - 30:
            break
        
        opt_pos, opt_val = optimize_positions(init_pos, max_time=20.0)
        
        if opt_val > best_value:
            best_value = opt_val
            best_positions = opt_pos
    
    # Final refinement with smaller steps
    if time.time() < deadline - 10:
        final_pos, final_val = optimize_positions(best_positions, max_time=15.0)
        if final_val > best_value:
            best_positions = final_pos
            best_value = final_val
    
    # Compute final radii
    radii = compute_radii(best_positions)
    
    # Build result array
    result = np.column_stack([best_positions, radii])
    
    return result
# EVOLVE-BLOCK-END
