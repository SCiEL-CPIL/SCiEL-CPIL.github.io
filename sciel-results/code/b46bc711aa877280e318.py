"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validate(arr):
    """Validate a packing solution. Returns True if valid."""
    if arr is None or arr.shape != (N, 3):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # Check radii are non-negative
    if np.any(r < 0):
        return False
    
    # Check all circles are within bounds (find bounding rectangle)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter constraint: 2(w+h) <= 4
    if 2 * (w + h) > 4 + 1e-9:
        return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - 1e-7:
                return False
    
    return True

def _compute_score(arr):
    """Compute the sum of radii for a valid packing."""
    if arr is None or not _validate(arr):
        return 0.0
    return np.sum(arr[:, 2])

def _soft_min(values, temperature=0.01):
    """Compute soft minimum using log-sum-exp trick."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    min_val = np.min(values)
    exp_vals = np.exp((values - min_val) / temperature)
    return min_val - temperature * np.log(np.sum(exp_vals))

def _soft_max(values, temperature=0.01):
    """Compute soft maximum using log-sum-exp trick."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    exp_vals = np.exp((values - max_val) / temperature)
    return max_val + temperature * np.log(np.sum(exp_vals))

def _physics_simulate(positions, radii, w, h, num_steps=500, start_temp=0.1, end_temp=0.001):
    """
    Physics-based soft-body simulation using Lennard-Jones-like potentials.
    Returns updated positions and radii.
    """
    positions = positions.copy()
    radii = radii.copy()
    
    # Lennard-Jones parameters
    sigma_factor = 1.0  # scale factor for sigma
    epsilon = 1.0  # energy scale
    
    # Boundary stiffness
    boundary_stiffness = 5.0
    
    for step in range(num_steps):
        if time.time() > 145.0:
            break
        
        # Annealing schedule
        temp = start_temp * (end_temp / start_temp) ** (step / num_steps)
        
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Inter-circle repulsive forces (Lennard-Jones-like)
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                
                # Effective sigma based on current radii
                sigma = sigma_factor * (radii[i] + radii[j])
                
                # Lennard-Jones potential: U = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
                # Force is -dU/dr = 4*epsilon*(12*sigma^12/r^13 - 6*sigma^6/r^7)
                ratio = sigma / dist
                if ratio > 2.0:
                    # Strong repulsion region
                    force_mag = epsilon * (12 * ratio**12 - 6 * ratio**6) / dist
                elif ratio > 0.5:
                    # Moderate interaction
                    force_mag = epsilon * (12 * ratio**12 - 6 * ratio**6) / dist
                else:
                    # Weak attraction (helps packing)
                    force_mag = epsilon * (12 * ratio**12 - 6 * ratio**6) / dist * 0.1
                
                fx = force_mag * dx / dist
                fy = force_mag * dy / dist
                
                forces[i, 0] -= fx
                forces[i, 1] -= fy
                forces[j, 0] += fx
                forces[j, 1] += fy
        
        # Boundary forces (soft walls)
        for i in range(N):
            r = radii[i]
            
            # Left wall
            dist_left = positions[i, 0] - r
            if dist_left < 0.1:
                forces[i, 0] += boundary_stiffness * dist_left
            
            # Right wall
            dist_right = w - positions[i, 0] - r
            if dist_right < 0.1:
                forces[i, 0] -= boundary_stiffness * dist_right
            
            # Bottom wall
            dist_bottom = positions[i, 1] - r
            if dist_bottom < 0.1:
                forces[i, 1] += boundary_stiffness * dist_bottom
            
            # Top wall
            dist_top = h - positions[i, 1] - r
            if dist_top < 0.1:
                forces[i, 1] -= boundary_stiffness * dist_top
        
        # Apply forces with temperature-based damping
        lr = 0.01 * temp
        positions += forces * lr
        
        # Add small random perturbation to escape local minima
        positions += np.random.randn(*positions.shape) * temp * 0.01
        
        # Clamp to valid region (hard constraint)
        for i in range(N):
            r = radii[i]
            positions[i, 0] = np.clip(positions[i, 0], r, w - r)
            positions[i, 1] = np.clip(positions[i, 1], r, h - r)
        
        # Dynamically adjust radii: try to expand each circle
        for i in range(N):
            # Max possible radius based on boundaries
            max_r_boundary = min(
                positions[i, 0], 
                w - positions[i, 0], 
                positions[i, 1], 
                h - positions[i, 1]
            )
            
            # Max radius based on neighbors
            max_r_neighbors = max_r_boundary
            for j in range(N):
                if i == j:
                    continue
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.hypot(dx, dy)
                avail = dist - radii[j]
                if avail < max_r_neighbors:
                    max_r_neighbors = avail
            
            # Smoothly update radius
            target_r = max(0.0, max_r_neighbors)
            radii[i] = 0.95 * radii[i] + 0.05 * target_r
    
    return positions, radii

def _polish(positions, radii, w, h, num_iterations=100):
    """High-precision polish pass using gradient-based optimization."""
    positions = positions.copy()
    radii = radii.copy()
    
    for it in range(num_iterations):
        if time.time() > 145.0:
            break
        
        # Compute exact max radii for current positions
        new_radii = np.zeros(N)
        for i in range(N):
            max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
            for j in range(N):
                if i == j:
                    continue
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.hypot(dx, dy)
                avail = dist - radii[j]
                if avail < max_r:
                    max_r = avail
            new_radii[i] = max(0.0, max_r)
        
        radii = new_radii
        
        # Compute forces to resolve any remaining overlaps
        forces = np.zeros_like(positions)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.hypot(dx, dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    if dist > 1e-10:
                        overlap = min_dist - dist
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
        
        # Apply small correction
        lr = 0.1 * (0.9 ** (it / 10.0))
        positions += forces * lr
        
        # Clamp
        for i in range(N):
            r = radii[i]
            positions[i, 0] = np.clip(positions[i, 0], r, w - r)
            positions[i, 1] = np.clip(positions[i, 1], r, h - r)
    
    return positions, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    # Perimeter 4 implies 2(w+h) = 4 => w+h = 2
    # We search for the optimal aspect ratio w/h
    aspect_ratios = [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0
    ]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        # Calculate w and h from aspect ratio and w+h=2
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Determine grid dimensions for initialization
        cols = max(1, int(np.round(np.sqrt(N * ar))))
        rows = max(1, int(np.ceil(N / cols)))
        
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        # Try multiple restarts with physics simulation
        num_restarts = 2
        for restart in range(num_restarts):
            if time.time() > deadline - 15.0:
                break
            
            # Initial positions: Grid layout with jitter
            positions = np.zeros((N, 2))
            for i in range(N):
                row = i // cols
                col = i % cols
                
                cell_w = w / cols
                cell_h = h / rows
                
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                
                # Add jitter for restart > 0
                if restart > 0:
                    jitter_scale = min(cell_w, cell_h) * 0.2
                    x += np.random.uniform(-jitter_scale, jitter_scale)
                    y += np.random.uniform(-jitter_scale, jitter_scale)
                
                # Clamp to boundaries
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)
                
                positions[i] = [x, y]
            
            # Initial radii: small
            min_cell_dim = min(w/cols, h/rows)
            radii = np.full(N, min_cell_dim * 0.2)
            
            # Run physics simulation
            positions, radii = _physics_simulate(positions, radii, w, h, num_steps=300)
            
            # Polish pass
            positions, radii = _polish(positions, radii, w, h, num_iterations=50)
            
            # Final Validation
            arr = np.column_stack([positions, radii])
            valid = _validate(arr)
            
            if valid:
                score = _compute_score(arr)
                if score > best_score:
                    best_score = score
                    best_result = arr.copy()
    
    # Fallback if no valid result was found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
