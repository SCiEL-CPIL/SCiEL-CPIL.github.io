"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Pack 21 circles in a variable rectangle of perimeter 4 to maximize sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Physics-based optimization
    np.random.seed(42)
    
    # Initial configuration: circles in a rough grid pattern
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Initialize in a grid pattern within a unit square
    for i in range(N):
        row = i // side
        col = i % side
        positions[i, 0] = (col + 0.5) / side
        positions[i, 1] = (row + 0.5) / side
        radii[i] = 0.1
    
    # Rectangle dimensions (width, height) - we'll optimize these too
    width = 1.0
    height = 1.0
    
    # Parameters for the simulation
    epsilon = 0.01  # Softness parameter for constraints
    alpha = 0.1     # Learning rate
    beta = 0.5      # Damping factor
    
    # Iterative optimization
    best_value = 0.0
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_width = width
    best_height = height
    
    for iteration in range(5000):
        # Check time limit
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(positions)
        radius_forces = np.zeros_like(radii)
        
        # Inter-circle repulsion (Lennard-Jones-like)
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Repulsive force
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    forces[i, 0] += force_mag * dx / dist if dist > 0 else 0
                    forces[i, 1] += force_mag * dy / dist if dist > 0 else 0
                    forces[j, 0] -= force_mag * dx / dist if dist > 0 else 0
                    forces[j, 1] -= force_mag * dy / dist if dist > 0 else 0
                    
                    # Encourage radii to decrease
                    radius_forces[i] -= 0.1 * overlap
                    radius_forces[j] -= 0.1 * overlap
        
        # Wall constraints
        for i in range(N):
            # Left wall
            if positions[i, 0] < radii[i]:
                forces[i, 0] += 10.0 * (radii[i] - positions[i, 0])
                radius_forces[i] -= 0.1 * (radii[i] - positions[i, 0])
            
            # Right wall
            if positions[i, 0] + radii[i] > width:
                forces[i, 0] -= 10.0 * (positions[i, 0] + radii[i] - width)
                radius_forces[i] -= 0.1 * (positions[i, 0] + radii[i] - width)
            
            # Bottom wall
            if positions[i, 1] < radii[i]:
                forces[i, 1] += 10.0 * (radii[i] - positions[i, 1])
                radius_forces[i] -= 0.1 * (radii[i] - positions[i, 1])
            
            # Top wall
            if positions[i, 1] + radii[i] > height:
                forces[i, 1] -= 10.0 * (positions[i, 1] + radii[i] - height)
                radius_forces[i] -= 0.1 * (positions[i, 1] + radii[i] - height)
        
        # Update positions and radii
        positions += alpha * forces * beta
        radii += alpha * radius_forces * beta
        
        # Ensure radii are positive
        radii = np.maximum(radii, 0.01)
        
        # Adjust rectangle dimensions to fit circles
        min_x = np.min(positions[:, 0] - radii)
        max_x = np.max(positions[:, 0] + radii)
        min_y = np.min(positions[:, 1] - radii)
        max_y = np.max(positions[:, 1] + radii)
        
        # Perimeter constraint: 2*(width + height) <= 4
        # So width + height <= 2
        current_width = max_x - min_x
        current_height = max_y - min_y
        
        if current_width + current_height > 2.0:
            # Scale down
            scale = 2.0 / (current_width + current_height)
            width *= scale
            height *= scale
            positions[:, 0] = min_x + (positions[:, 0] - min_x) * scale
            positions[:, 1] = min_y + (positions[:, 1] - min_y) * scale
            radii *= scale
        else:
            width = current_width
            height = current_height
        
        # Compute objective value
        value = np.sum(radii)
        
        # Track best
        if value > best_value:
            best_value = value
            best_positions = positions.copy()
            best_radii = radii.copy()
            best_width = width
            best_height = height
        
        # Decay learning rate
        alpha *= 0.999
    
    # Final validation and cleanup
    # Ensure all circles are within bounds and non-overlapping
    for i in range(N):
        # Clamp to bounds
        best_positions[i, 0] = np.clip(best_positions[i, 0], best_radii[i], best_width - best_radii[i])
        best_positions[i, 1] = np.clip(best_positions[i, 1], best_radii[i], best_height - best_radii[i])
    
    # Build result array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_positions[i, 0]
        result[i, 1] = best_positions[i, 1]
        result[i, 2] = best_radii[i]
    
    return result
# EVOLVE-BLOCK-END
