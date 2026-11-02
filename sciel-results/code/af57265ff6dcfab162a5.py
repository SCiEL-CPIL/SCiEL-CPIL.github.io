"""Circle packing in unit square - physics-based optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Pack 26 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initial configuration: grid layout
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([0.5 / side, 0.5 / side, 0.5 / side - 1e-4])
    # Use first N cells
    cells = cells[:N]
    
    # Convert to numpy array
    positions = np.array([c[:2] for c in cells])
    radii = np.array([c[2] for c in cells])
    
    # Best solution tracking
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Optimization parameters
    lr_pos = 0.01
    lr_rad = 0.001
    overlap_penalty_weight = 10.0
    boundary_penalty_weight = 50.0
    
    # Gradient descent with momentum
    vel_pos = np.zeros_like(positions)
    vel_rad = np.zeros_like(radii)
    momentum = 0.9
    
    for iteration in range(50000):
        # Check deadline
        if time.time() > deadline - 5:
            break
        
        # Compute forces/gradients
        grad_pos = np.zeros_like(positions)
        grad_rad = np.zeros_like(radii)
        
        # Pairwise overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq + 1e-10)
                
                min_dist = radii[i] + radii[j]
                overlap = min_dist - dist
                
                if overlap > 0:
                    # Soft penalty: quadratic in overlap
                    force_mag = 2.0 * overlap * overlap_penalty_weight
                    if dist > 1e-8:
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                    else:
                        # Random direction if coincident
                        angle = np.random.uniform(0, 2 * np.pi)
                        fx = force_mag * np.cos(angle)
                        fy = force_mag * np.sin(angle)
                    
                    grad_pos[i, 0] -= fx
                    grad_pos[i, 1] -= fy
                    grad_pos[j, 0] += fx
                    grad_pos[j, 1] += fy
                    
                    # Radius gradient: reduce radii to reduce overlap
                    grad_rad[i] += overlap * overlap_penalty_weight
                    grad_rad[j] += overlap * overlap_penalty_weight
        
        # Boundary constraints: circles must stay inside [0,1]^2
        for i in range(N):
            # Left boundary
            overlap_left = radii[i] - positions[i, 0]
            if overlap_left > 0:
                grad_pos[i, 0] += overlap_left * boundary_penalty_weight
                grad_rad[i] += overlap_left * boundary_penalty_weight
            
            # Right boundary
            overlap_right = radii[i] - (1.0 - positions[i, 0])
            if overlap_right > 0:
                grad_pos[i, 0] -= overlap_right * boundary_penalty_weight
                grad_rad[i] += overlap_right * boundary_penalty_weight
            
            # Bottom boundary
            overlap_bottom = radii[i] - positions[i, 1]
            if overlap_bottom > 0:
                grad_pos[i, 1] += overlap_bottom * boundary_penalty_weight
                grad_rad[i] += overlap_bottom * boundary_penalty_weight
            
            # Top boundary
            overlap_top = radii[i] - (1.0 - positions[i, 1])
            if overlap_top > 0:
                grad_pos[i, 1] -= overlap_top * boundary_penalty_weight
                grad_rad[i] += overlap_top * boundary_penalty_weight
        
        # Objective: maximize sum of radii (minimize negative sum)
        grad_rad -= 1.0  # Gradient of -sum(radii) is -1 for each radius
        
        # Update with momentum
        vel_pos = momentum * vel_pos + lr_pos * grad_pos
        vel_rad = momentum * vel_rad + lr_rad * grad_rad
        
        # Apply updates
        positions += vel_pos
        radii += vel_rad
        
        # Ensure radii are positive
        radii = np.maximum(radii, 1e-6)
        
        # Clamp positions to valid range
        positions[:, 0] = np.clip(positions[:, 0], 0, 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, 1)
        
        # Check if this is a valid solution (no overlaps, inside bounds)
        valid = True
        for i in range(N):
            if positions[i, 0] < radii[i] or positions[i, 0] > 1 - radii[i]:
                valid = False
                break
            if positions[i, 1] < radii[i] or positions[i, 1] > 1 - radii[i]:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[i] + radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # Adaptive learning rate: reduce if oscillating
        if iteration % 1000 == 0 and iteration > 0:
            lr_pos *= 0.95
            lr_rad *= 0.95
    
    # Final polish: try to increase radii slightly while maintaining validity
    for polish_iter in range(1000):
        if time.time() > deadline - 2:
            break
        
        # Try to increase all radii by a small amount
        delta_r = 1e-5
        new_radii = best_radii + delta_r
        
        # Check if still valid
        valid = True
        for i in range(N):
            if best_positions[i, 0] < new_radii[i] or best_positions[i, 0] > 1 - new_radii[i]:
                valid = False
                break
            if best_positions[i, 1] < new_radii[i] or best_positions[i, 1] > 1 - new_radii[i]:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = best_positions[i, 0] - best_positions[j, 0]
                    dy = best_positions[i, 1] - best_positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            best_radii = new_radii
        else:
            # Try smaller increment
            delta_r *= 0.5
            if delta_r < 1e-8:
                break
    
    # Construct final result
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = best_radii
    
    return result
# EVOLVE-BLOCK-END
