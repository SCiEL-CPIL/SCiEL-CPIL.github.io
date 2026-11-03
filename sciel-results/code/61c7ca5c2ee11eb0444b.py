"""Physics-based circle packing in perimeter-4 rectangle."""
import numpy as np
import random

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Simulated Annealing with vectorized overlap checks.
    """
    best_score = -np.inf
    best_result = None
    
    # Try multiple random restarts with different initial aspect ratios
    for restart in range(10):
        # Randomize initial aspect ratio slightly
        # w + h = 2. Let w = 2 * t, h = 2 * (1-t) where t in [0.2, 0.8]
        t = random.uniform(0.2, 0.8)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        
        # Initialize positions and radii
        # Grid-like initialization to avoid immediate massive overlaps
        side = int(np.ceil(np.sqrt(N)))
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            # Add small random jitter to break symmetry
            positions[i, 0] = (col + 0.5) * w / side + random.uniform(-0.02, 0.02)
            positions[i, 1] = (row + 0.5) * h / side + random.uniform(-0.02, 0.02)
            radii[i] = 0.05 # Small initial radius
            
        # Ensure initial positions are within bounds (roughly)
        # We will rely on the annealing to fix this, but let's clamp
        positions[:, 0] = np.clip(positions[:, 0], 0, w)
        positions[:, 1] = np.clip(positions[:, 1], 0, h)
        
        result = _simulated_annealing(positions, radii, w, h, N, max_iters=50000)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
                
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) * w / side
            y = (row + 0.5) * h / side
            r = min(w, h) / (2 * side) * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

def _simulated_annealing(pos, rad, w, h, N, max_iters=50000):
    """
    Simulated Annealing to maximize sum(radii) subject to non-overlap and boundary constraints.
    Variables: pos (N,2), rad (N,), w, h (with w+h=2).
    """
    # State
    current_pos = pos.copy()
    current_rad = rad.copy()
    current_w = w
    current_h = h
    
    # Calculate initial energy
    # Energy = -sum(rad) + penalty
    # We want to minimize Energy.
    
    def calculate_energy(p, r, w_curr, h_curr):
        # Boundary violations
        # For each circle, distance to nearest boundary must be >= r
        # x in [r, w-r], y in [r, h-r]
        
        # Vectorized boundary penalty
        # Left: max(0, r - x)
        # Right: max(0, x - (w-r))
        # Bottom: max(0, r - y)
        # Top: max(0, y - (h-r))
        
        left_viol = np.maximum(0, r - p[:, 0])
        right_viol = np.maximum(0, p[:, 0] - (w_curr - r))
        bottom_viol = np.maximum(0, r - p[:, 1])
        top_viol = np.maximum(0, p[:, 1] - (h_curr - r))
        
        boundary_pen = np.sum(left_viol**2 + right_viol**2 + bottom_viol**2 + top_viol**2)
        
        # Overlap penalty
        # Pairwise distances
        # d_ij = ||p_i - p_j||
        # violation = max(0, r_i + r_j - d_ij)
        
        # Use broadcasting for pairwise differences
        # diff shape (N, N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Sum of radii matrix
        rad_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Violations
        viol = np.maximum(0, rad_sum - dists)
        
        # Only count upper triangle to avoid double counting, or just sum all and divide by 2?
        # Sum of squares of violations
        overlap_pen = 0.5 * np.sum(viol**2)
        
        # Objective: Maximize sum(r) => Minimize -sum(r) + Penalty
        # We scale penalty to be significant
        return -np.sum(r) + 10.0 * (boundary_pen + overlap_pen)
    
    current_energy = calculate_energy(current_pos, current_rad, current_w, current_h)
    
    # Annealing parameters
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.9995
    
    best_pos = current_pos.copy()
    best_rad = current_rad.copy()
    best_w = current_w
    best_h = current_h
    best_energy = current_energy
    
    for i in range(max_iters):
        T *= cooling_rate
        
        # Generate neighbor
        new_pos = current_pos.copy()
        new_rad = current_rad.copy()
        new_w = current_w
        new_h = current_h
        
        # 1. Move a random circle
        idx = random.randint(0, N - 1)
        step_size = 0.05 * (1 + T) # Scale step with temperature
        new_pos[idx, 0] += random.gauss(0, step_size)
        new_pos[idx, 1] += random.gauss(0, step_size)
        
        # 2. Change radius of a random circle
        idx_r = random.randint(0, N - 1)
        rad_step = 0.01 * (1 + T)
        new_rad[idx_r] += random.gauss(0, rad_step)
        # Keep radius positive
        if new_rad[idx_r] < 0.01:
            new_rad[idx_r] = 0.01
            
        # 3. Change aspect ratio (width)
        # w + h = 2. Let w = 2 * sigmoid(u) or just adjust w directly
        # Adjust w by small amount
        dw = random.gauss(0, 0.02 * (1 + T))
        new_w = current_w + dw
        # Constrain w to reasonable range [0.2, 1.8]
        if new_w < 0.2: new_w = 0.2
        if new_w > 1.8: new_w = 1.8
        new_h = 2.0 - new_w
        
        # Calculate new energy
        new_energy = calculate_energy(new_pos, new_rad, new_w, new_h)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0:
            # Improvement
            current_pos = new_pos
            current_rad = new_rad
            current_w = new_w
            current_h = new_h
            current_energy = new_energy
        else:
            # Accept with probability exp(-delta_e / T)
            if random.random() < np.exp(-delta_e / T):
                current_pos = new_pos
                current_rad = new_rad
                current_w = new_w
                current_h = new_h
                current_energy = new_energy
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            best_rad = current_rad.copy()
            best_w = current_w
            best_h = current_h
            
    # Final verification and cleanup
    # We need to ensure the best solution is valid.
    # The energy function includes penalties, so low energy implies low violation.
    # However, we should verify strict validity for the return value.
    
    # Check validity of best_pos, best_rad, best_w, best_h
    valid = True
    p = best_pos
    r = best_rad
    w_curr = best_w
    h_curr = best_h
    
    # Boundary check
    for i in range(N):
        x, y, rad_i = p[i, 0], p[i, 1], r[i]
        if x < rad_i - 1e-5 or x > w_curr - rad_i + 1e-5 or y < rad_i - 1e-5 or y > h_curr - rad_i + 1e-5:
            valid = False
            break
            
    if valid:
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < r[i] + r[j] - 1e-5:
                    valid = False
                    break
            if not valid:
                break
    
    if valid:
        result = np.zeros((N, 3))
        for i in range(N):
            result[i, 0] = p[i, 0]
            result[i, 1] = p[i, 1]
            result[i, 2] = r[i]
        return result
    else:
        # If invalid, try to repair by shrinking radii? 
        # Or just return None to trigger next restart
        return None

# EVOLVE-BLOCK-END
