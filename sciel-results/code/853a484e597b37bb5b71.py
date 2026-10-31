"""Circle packing in unit square: force-directed relaxation approach."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    np.random.seed(42)
    
    # Initial configuration: place circles in a rough grid with some jitter
    # For 26 circles, use a 6x5 grid (30 cells) and take 26
    cols = 6
    rows = 5
    x_coords = np.linspace(0.08, 0.92, cols)
    y_coords = np.linspace(0.08, 0.92, rows)
    
    positions = []
    for j in range(rows):
        for i in range(cols):
            if len(positions) < N:
                positions.append([x_coords[i] + np.random.uniform(-0.02, 0.02),
                                  y_coords[j] + np.random.uniform(-0.02, 0.02)])
    positions = np.array(positions[:N], dtype=float)
    
    # Initial radii: start with small circles and grow them
    radii = np.full(N, 0.05)
    
    # Force-directed relaxation
    # Parameters
    repulsion_strength = 1.0
    attraction_strength = 0.1
    damping = 0.9
    dt = 0.01
    
    best_score = -np.inf
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
            
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion: if circles overlap, push apart
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                
                if dist < 1e-10:
                    # Random small perturbation
                    diff = np.random.uniform(-0.01, 0.01, 2)
                    dist = np.linalg.norm(diff)
                
                if dist < min_dist:
                    # Overlapping: strong repulsion
                    overlap = min_dist - dist
                    force_magnitude = repulsion_strength * overlap
                    direction = diff / dist
                    forces[i] += direction * force_magnitude
                    forces[j] -= direction * force_magnitude
                elif dist < min_dist * 2.0:
                    # Near contact: mild attraction to maintain contact
                    force_magnitude = attraction_strength * (dist - min_dist)
                    direction = diff / dist
                    forces[i] -= direction * force_magnitude
                    forces[j] += direction * force_magnitude
        
        # Wall forces: keep circles inside [0,1]^2
        for i in range(N):
            r = radii[i]
            # Left wall
            if positions[i, 0] < r:
                forces[i, 0] += repulsion_strength * (r - positions[i, 0])
            # Right wall
            if positions[i, 0] > 1.0 - r:
                forces[i, 0] -= repulsion_strength * (positions[i, 0] - (1.0 - r))
            # Bottom wall
            if positions[i, 1] < r:
                forces[i, 1] += repulsion_strength * (r - positions[i, 1])
            # Top wall
            if positions[i, 1] > 1.0 - r:
                forces[i, 1] -= repulsion_strength * (positions[i, 1] - (1.0 - r))
        
        # Update positions
        velocities = forces * dt
        positions += velocities
        positions *= damping  # damping to stabilize
        
        # Keep within bounds (hard constraint)
        for i in range(N):
            r = radii[i]
            positions[i, 0] = np.clip(positions[i, 0], r, 1.0 - r)
            positions[i, 1] = np.clip(positions[i, 1], r, 1.0 - r)
        
        # Gradually increase radii
        if iteration % 50 == 0:
            radii *= 1.005
        
        # Check validity and score
        valid = True
        for i in range(N):
            r = radii[i]
            if positions[i, 0] < -1e-9 or positions[i, 0] > 1.0 + 1e-9:
                valid = False
                break
            if positions[i, 1] < -1e-9 or positions[i, 1] > 1.0 + 1e-9:
                valid = False
                break
            if r < 0:
                valid = False
                break
        
        if valid:
            # Check pairwise distances
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # If too many overlaps, shrink radii
        if not valid:
            radii *= 0.95
    
    # Final result: use best valid configuration found
    # Ensure all circles are strictly inside
    final_positions = best_positions.copy()
    final_radii = best_radii.copy()
    
    # Shrink radii slightly to ensure validity
    for i in range(N):
        r = final_radii[i]
        # Check if circle is fully inside
        margin = min(final_positions[i, 0], 1.0 - final_positions[i, 0],
                     final_positions[i, 1], 1.0 - final_positions[i, 1])
        if r > margin:
            final_radii[i] = margin * 0.999
    
    # Final validity check
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(final_positions[i] - final_positions[j])
            if dist < final_radii[i] + final_radii[j]:
                # Shrink the smaller radius
                if final_radii[i] < final_radii[j]:
                    final_radii[i] = max(0, (dist - final_radii[j]) * 0.999)
                else:
                    final_radii[j] = max(0, (dist - final_radii[i]) * 0.999)
    
    # Build output array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = final_positions[i, 0]
        result[i, 1] = final_positions[i, 1]
        result[i, 2] = final_radii[i]
    
    return result
# EVOLVE-BLOCK-END
