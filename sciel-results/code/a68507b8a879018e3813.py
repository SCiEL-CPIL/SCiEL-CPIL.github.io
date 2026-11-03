import numpy as np

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def calculate_radii(positions, max_iters=20):
    """
    Calculate the maximum feasible radii for given positions.
    Uses iterative relaxation.
    """
    n = len(positions)
    radii = np.zeros(n)
    # Initial guess: distance to nearest wall
    for i in range(n):
        x, y = positions[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)
    
    # Iterate to resolve overlaps
    for _ in range(max_iters):
        for i in range(n):
            x, y = positions[i]
            max_r = min(x, y, 1.0 - x, 1.0 - y)
            for j in range(n):
                if i == j:
                    continue
                dx = x - positions[j][0]
                dy = y - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # The radius of i must be such that dist >= r_i + r_j
                # So r_i <= dist - r_j
                candidate_r = dist - radii[j]
                if candidate_r < max_r:
                    max_r = candidate_r
            # Ensure radius is non-negative
            radii[i] = max(0.0, max_r)
            
    return radii

def construct():
    # Set random seed for reproducibility in testing, but usually we want randomness for diversity
    # random.seed(42) 
    
    # Initial random positions
    positions = np.random.rand(N, 2)
    
    # Calculate initial radii
    radii = calculate_radii(positions)
    current_score = np.sum(radii)
    
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    steps = 50000  # Number of steps
    
    # Step size for movement
    step_size = 0.05
    
    for step in range(steps):
        # Pick a random circle to move
        i = random.randint(0, N - 1)
        
        # Generate a random perturbation
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        
        new_x = positions[i][0] + dx
        new_y = positions[i][1] + dy
        
        # Keep inside the square
        if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
            continue
            
        # Save old position
        old_x, old_y = positions[i]
        
        # Move the circle
        positions[i] = [new_x, new_y]
        
        # Recalculate radii
        new_radii = calculate_radii(positions)
        new_score = np.sum(new_radii)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            # Accept if better
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_positions = positions.copy()
                best_radii = new_radii.copy()
        else:
            # Accept if worse with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                current_score = new_score
            else:
                # Revert move
                positions[i] = [old_x, old_y]
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally restart with higher T or smaller step size if stuck
            # For now, just continue with min T
            
    # Finalize: Ensure we return the best found
    # The best radii correspond to best_positions
    # Re-calculate radii for best_positions to be sure (though best_radii should be correct)
    final_radii = calculate_radii(best_positions)
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_positions[i][0], best_positions[i][1], final_radii[i]]
        
    return result
# EVOLVE-BLOCK-END
