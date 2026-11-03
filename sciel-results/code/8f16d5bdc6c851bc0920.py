"""Physics-based circle packing in perimeter-4 rectangle using iterative relaxation."""
import numpy as np

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Physics-based iterative relaxation with repulsive forces.
    - Initialize circles in a grid with small radii
    - Iteratively: apply repulsive forces between overlapping circles,
      boundary constraints, and gradually increase target radius
    - Use multiple restarts with different aspect ratios and initial configs
    """
    best_result = None
    best_score = -np.inf
    
    # Try multiple aspect ratios
    aspects = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5]
    
    for aspect in aspects:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Multiple initial configurations
        for init_type in ['grid', 'random1', 'random2']:
            result = _physics_pack(w, h, N, init_type=init_type)
            if result is not None:
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
    
    if best_result is None:
        # Fallback: simple grid in square
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cx = (col + 0.5) * w / side
            cy = (row + 0.5) * h / side
            r = min(w / side, h / side) * 0.45
            cells.append([cx, cy, r])
        return np.array(cells)
    
    return best_result


def _physics_pack(w, h, N, init_type='grid', max_iters=2000, dt=0.005):
    """
    Physics-based packing for fixed rectangle w x h (w+h=2).
    Uses repulsive forces and boundary constraints.
    """
    # Initialize positions and radii
    positions = np.zeros((N, 2))
    
    if init_type == 'grid':
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            row = i // side
            col = i % side
            positions[i, 0] = (col + 0.5) * w / side
            positions[i, 1] = (row + 0.5) * h / side
    elif init_type == 'random1':
        np.random.seed(42)
        positions[:, 0] = np.random.uniform(0.1*w, 0.9*w, N)
        positions[:, 1] = np.random.uniform(0.1*h, 0.9*h, N)
    elif init_type == 'random2':
        np.random.seed(123)
        positions[:, 0] = np.random.uniform(0.05*w, 0.95*w, N)
        positions[:, 1] = np.random.uniform(0.05*h, 0.95*h, N)
    
    # Start with small radii
    min_dim = min(w, h)
    initial_r = min_dim / (2 * int(np.ceil(np.sqrt(N)))) * 0.8
    radii = np.full(N, initial_r)
    
    # Target radius: we'll gradually increase it
    # Upper bound: each circle can be at most min(w,h)/2
    max_possible_r = min(w, h) / 2.0
    
    # We'll try to reach a target radius and see if it's feasible
    # Strategy: start small, increase target, relax until no overlaps, then increase
    
    target_r = initial_r * 1.2
    current_r = initial_r
    
    for it in range(max_iters):
        # Gradually increase target radius
        if it < 500:
            # Ramp up target
            target_r = initial_r + (max_possible_r * 0.85 - initial_r) * (it / 500.0)
        else:
            target_r = max_possible_r * 0.85
        
        # Update current radii towards target
        current_r = np.full(N, target_r)
        
        # Apply forces
        forces = np.zeros((N, 2))
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = current_r[i] + current_r[j]
                
                if dist < min_dist and dist > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force_mag = overlap * 10.0  # stiffness
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist <= 1e-10:
                    # Random small push to avoid singularity
                    angle = np.random.uniform(0, 2*np.pi)
                    forces[i, 0] += 0.01 * np.cos(angle)
                    forces[i, 1] += 0.01 * np.sin(angle)
                    forces[j, 0] -= 0.01 * np.cos(angle)
                    forces[j, 1] -= 0.01 * np.sin(angle)
        
        # Boundary forces: keep circles inside rectangle
        for i in range(N):
            x, y = positions[i]
            r = current_r[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            # Right wall
            if x > w - r:
                forces[i, 0] -= (x - (w - r)) * 10.0
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            # Top wall
            if y > h - r:
                forces[i, 1] -= (y - (h - r)) * 10.0
        
        # Update positions
        positions += forces * dt
        
        # Clamp positions to stay within bounds (hard constraint)
        for i in range(N):
            r = current_r[i]
            positions[i, 0] = np.clip(positions[i, 0], r, w - r)
            positions[i, 1] = np.clip(positions[i, 1], r, h - r)
        
        # Check if we've converged (no significant forces)
        if it > 100:
            force_mag = np.max(np.linalg.norm(forces, axis=1))
            if force_mag < 1e-6:
                break
    
    # Final verification and cleanup
    # Shrink radii slightly to ensure no overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = current_r[i] + current_r[j]
            if dist < min_dist:
                # Shrink both radii
                shrink = (min_dist - dist) / 2.0 + 1e-8
                current_r[i] -= shrink
                current_r[j] -= shrink
        
        # Ensure inside bounds
        r = current_r[i]
        if r < 1e-6:
            r = 1e-6
        positions[i, 0] = np.clip(positions[i, 0], r, w - r)
        positions[i, 1] = np.clip(positions[i, 1], r, h - r)
        current_r[i] = r
    
    # Final check: verify no overlaps and all inside
    valid = True
    for i in range(N):
        x, y, r = positions[i, 0], positions[i, 1], current_r[i]
        if r <= 0 or x < r - 1e-6 or x > w - r + 1e-6 or y < r - 1e-6 or y > h - r + 1e-6:
            valid = False
            break
    
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < current_r[i] + current_r[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if valid:
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [positions[i, 0], positions[i, 1], current_r[i]]
        return result
    
    return None

# EVOLVE-BLOCK-END
