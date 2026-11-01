"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii.
    Uses Simulated Annealing with repulsion forces.
    """
    np.random.seed(42) # For reproducibility in debugging, remove for production randomness if needed
    
    # Initialize positions and radii
    # Start with a hexagonal-like grid to provide a good initial structure
    side = int(np.ceil(np.sqrt(N)))
    dx = 1.0 / side
    dy = 1.0 / side
    
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            # Offset every other row to create hexagonal packing
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            if i % 2 == 1:
                x += dx * 0.5
            
            # Keep within bounds roughly
            x = np.clip(x, 0.1, 0.9)
            y = np.clip(y, 0.1, 0.9)
            
            positions[count] = [x, y]
            radii[count] = dx * 0.4 # Initial small radius
            count += 1
            
    # Fill remaining if grid didn't cover N exactly (unlikely for 32 with side 6)
    while count < N:
        positions[count] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
        radii[count] = 0.05
        count += 1
        
    # Optimization parameters
    T = 1.0 # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-6
    max_iter = 2000
    alpha = 0.1 # Learning rate base
    
    best_sum_r = np.sum(radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    def compute_forces(pos, rad):
        forces = np.zeros_like(pos)
        radial_forces = np.zeros_like(rad)
        
        n = len(rad)
        
        # 1. Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = pos[i] - pos[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    # Overlap detected
                    if dist > 1e-10:
                        force_dir = dist_vec / dist
                    else:
                        force_dir = np.array([1.0, 0.0]) # Random direction if coincident
                    
                    # Force magnitude proportional to overlap
                    overlap = min_dist - dist
                    # Push apart
                    forces[i] += force_dir * overlap
                    forces[j] -= force_dir * overlap
                    
                    # Shrink radii slightly to resolve overlap faster
                    radial_forces[i] -= overlap * 0.5
                    radial_forces[j] -= overlap * 0.5
                else:
                    # No overlap: try to expand radii
                    # Only expand if not touching boundary or other circles too closely
                    # Simple heuristic: expand if space is available
                    space_x = min(pos[i][0] - rad[i], 1.0 - pos[i][0] - rad[i])
                    space_y = min(pos[i][1] - rad[i], 1.0 - pos[i][1] - rad[i])
                    # Check nearest neighbor distance roughly
                    # For efficiency, we just assume if no overlap, we can grow a bit
                    radial_forces[i] += 0.01
                    radial_forces[j] += 0.01
        
        # 2. Boundary containment
        for i in range(n):
            x, y = pos[i]
            r = rad[i]
            
            # Left wall
            if x - r < 0:
                forces[i][0] += (x - r) # Push right (positive x)
                radial_forces[i] += (x - r) * 0.5 # Shrink or move? Pushing is better
            # Right wall
            if x + r > 1:
                forces[i][0] -= (x + r - 1) # Push left (negative x)
                radial_forces[i] += (x + r - 1) * 0.5
            # Bottom wall
            if y - r < 0:
                forces[i][1] += (y - r) # Push up
                radial_forces[i] += (y - r) * 0.5
            # Top wall
            if y + r > 1:
                forces[i][1] -= (y + r - 1) # Push down
                radial_forces[i] += (y + r - 1) * 0.5
            
            # Ensure radius is non-negative
            if r < 0:
                radial_forces[i] += 1.0
            
        return forces, radial_forces

    for it in range(max_iter):
        forces, radial_forces = compute_forces(positions, radii)
        
        # Update positions and radii
        # Learning rate decays with temperature
        lr = alpha * T
        
        # Add noise (Simulated Annealing step)
        noise = np.random.randn(*positions.shape) * T * 0.1
        r_noise = np.random.randn(N) * T * 0.01
        
        positions += forces * lr + noise
        radii += radial_forces * lr + r_noise
        
        # Clip positions to [0,1] and radii to [0, 0.5]
        positions = np.clip(positions, 0.0, 1.0)
        radii = np.clip(radii, 0.0, 0.5)
        
        # Check validity and update best
        # Simple validity check: no overlaps, inside box
        valid = True
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
                valid = False
                break
            if r < -1e-7:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < radii[i] + radii[j] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current_sum = np.sum(radii)
            if current_sum > best_sum_r:
                best_sum_r = current_sum
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            break
            
    # Final cleanup: ensure strict validity
    # If the best found is invalid (due to floating point errors in check), we might need to shrink radii slightly
    # But our check was strict. Let's return the best found.
    
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = best_radii
    
    return result
# EVOLVE-BLOCK-END
