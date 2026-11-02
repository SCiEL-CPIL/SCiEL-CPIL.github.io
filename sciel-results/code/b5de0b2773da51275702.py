"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def hex_support(theta, phi):
        """Support function of unit hexagon (circumradius 1) rotated by theta in direction phi."""
        angles = theta + 2 * np.pi * np.arange(6) / 6.0
        return np.max(np.cos(angles - phi))
    
    def compute_required_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation=0.0):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        ocx, ocy = outer_center
        beta = outer_rotation
        # 6 normal directions of outer hexagon
        phi_dirs = beta + np.pi / 6.0 + np.pi * np.arange(6) / 3.0
        
        max_extent = 0.0
        for i in range(len(positions)):
            x, y = positions[i]
            theta = angles[i]
            for phi in phi_dirs:
                # Extent of inner hex i in direction phi, relative to outer center
                extent = (x - ocx) * np.cos(phi) + (y - ocy) * np.sin(phi) + hex_support(theta, phi)
                if extent > max_extent:
                    max_extent = extent
        
        inradius = max_extent
        side_length = inradius * 2.0 / math.sqrt(3.0)
        return side_length
    
    # Initial placement: hexagonal packing pattern
    # 12 hexagons: center + ring of 6 + partial ring
    # Let's use a compact arrangement
    # For N=12, try: 1 center, 6 in first ring, 5 in second ring (partial)
    
    def initial_placement():
        """Generate initial positions for 12 unit hexagons."""
        positions = []
        angles = []
        
        # Center hexagon
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        # First ring: 6 hexagons at distance ~2 (hexagons touching)
        # For hexagons with circumradius 1, touching distance is 2
        r1 = 2.0
        for k in range(6):
            angle = 2 * np.pi * k / 6.0
            positions.append([r1 * np.cos(angle), r1 * np.sin(angle)])
            angles.append(0.0)
        
        # Second ring: 5 more hexagons
        r2 = 4.0
        for k in range(5):
            angle = 2 * np.pi * k / 6.0 + np.pi / 6.0
            positions.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            angles.append(0.0)
        
        return np.array(positions), np.array(angles)
    
    def optimize_from_initial(pos0, ang0, max_iters=5000):
        """Local optimization using coordinate descent / Nelder-Mead style."""
        positions = pos0.copy()
        angles = ang0.copy()
        best_side = compute_required_side(positions, angles)
        
        # Use simple hill-climbing with random perturbations
        np.random.seed(42)
        step_size = 0.1
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try perturbing a random inner hexagon
            idx = np.random.randint(N)
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            dtheta = np.random.uniform(-step_size, step_size)
            
            old_x, old_y = positions[idx, 0], positions[idx, 1]
            old_theta = angles[idx]
            
            positions[idx, 0] = old_x + dx
            positions[idx, 1] = old_y + dy
            angles[idx] = old_theta + dtheta
            
            side = compute_required_side(positions, angles)
            
            if side < best_side:
                best_side = side
                no_improve = 0
            else:
                positions[idx, 0] = old_x
                positions[idx, 1] = old_y
                angles[idx] = old_theta
                no_improve += 1
                if no_improve > 50:
                    step_size *= 0.95
                    no_improve = 0
            
            if step_size < 1e-6:
                break
        
        return positions, angles, best_side
    
    # Run multiple random restarts
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    n_restarts = 0
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate perturbed initial placement
        pos0, ang0 = initial_placement()
        # Add small random perturbation
        pos0 += np.random.uniform(-0.2, 0.2, size=pos0.shape)
        ang0 += np.random.uniform(-0.1, 0.1, size=N)
        
        positions, angles, side = optimize_from_initial(pos0, ang0, max_iters=2000)
        
        if side < best_side:
            best_side = side
            best_positions = positions.copy()
            best_angles = angles.copy()
        
        n_restarts += 1
    
    if best_positions is None:
        # Fallback to initial placement
        best_positions, best_angles = initial_placement()
        best_side = compute_required_side(best_positions, best_angles)
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(math.degrees(best_angles[i]))])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
# EVOLVE-BLOCK-END
