"""Circle packing in perimeter-4 rectangle: Simulated Annealing approach."""
import numpy as np
import time
import random

N = 21

def _energy(x, y, r, w, h, penalty_weight=100.0):
    """
    Calculate the energy of the configuration.
    Energy = -sum(r) + penalty_weight * (sum of overlaps squared) + penalty_weight * (sum of boundary violations squared)
    Lower is better.
    """
    if w < 0.1 or h < 0.1:
        return 1e6
    
    # Boundary violations
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    
    boundary_pen = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )
    
    # Overlap violations
    # Vectorized calculation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    overlap_pen = penalty_weight * np.sum(overlap * overlap)
    
    return -np.sum(r) + boundary_pen + overlap_pen

def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if the configuration is strictly valid."""
    if w <= 0.01 or h <= 0.01:
        return False
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    # Check overlaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    if np.any(d_sq_pairs < r_sum_pairs * r_sum_pairs - tol):
        return False
    return True

def _expand_radii(x, y, r, w, h, max_iters=200):
    """
    Greedily expand radii as much as possible while maintaining validity.
    This is a local search step that helps maximize the objective once a valid layout is found.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Calculate max possible radius for circle i
            # Limited by walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            
            # Limited by other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_limit = d - r[j]
                if r_limit < r_max:
                    r_max = r_limit
            
            r_max = max(0.0, r_max)
            
            if r_max > r[i] + 1e-7:
                # Try to increase radius
                new_r = r_max
                r[i] = new_r
                if _is_valid(x, y, r, w, h):
                    improved = True
                else:
                    r[i] = r[i] # revert if it broke validity (shouldn't happen if r_max was calculated correctly, but safe)
                    # Actually, if r_max is calculated from current positions, setting r[i]=r_max should be valid
                    # unless floating point errors.
        
        if not improved:
            break
            
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_value = 0.0
    best_result = None
    
    rng = np.random.default_rng(42)
    
    # Parameters for Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.995
    
    # We will run multiple restarts with different initializations
    for restart in range(10):
        if time.time() > deadline - 30:
            break
            
        # Initialization: Random valid-ish start
        # Try different aspect ratios
        w = rng.uniform(0.6, 1.4)
        h = 2.0 - w
        
        # Start with small radii to avoid initial overlaps
        x = rng.uniform(0.1, w - 0.1, N)
        y = rng.uniform(0.1, h - 0.1, N)
        r = np.full(N, 0.05)
        
        # Initial energy
        current_energy = _energy(x, y, r, w, h, penalty_weight=100.0)
        
        T = T_start
        steps = 0
        max_steps = 5000
        stagnation_counter = 0
        best_local_energy = current_energy
        best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
        
        while T > T_end and steps < max_steps:
            steps += 1
            
            # Choose a move
            move_type = rng.integers(0, 3)
            
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            
            if move_type == 0:
                # Move a circle
                i = rng.integers(0, N)
                dx = rng.normal(0, 0.05)
                dy = rng.normal(0, 0.05)
                new_x[i] += dx
                new_y[i] += dy
                # Clamp to bounds roughly to keep energy sane
                new_x[i] = np.clip(new_x[i], 0.01, w - 0.01)
                new_y[i] = np.clip(new_y[i], 0.01, h - 0.01)
                
            elif move_type == 1:
                # Change radius of a circle
                i = rng.integers(0, N)
                factor = 1.0 + rng.normal(0, 0.1)
                new_r[i] = max(0.01, r[i] * factor)
                
            else:
                # Swap radii of two circles (helps with symmetry breaking)
                i, j = rng.choice(N, size=2, replace=False)
                new_r[i], new_r[j] = r[j], r[i]
                # Also swap positions? No, just radii is often enough for packing
                # Or move one towards the other? Let's stick to radius swap + position move.
                # Let's add a "nudge" move: move two circles towards each other or away
                if rng.random() < 0.5:
                     new_x[i] += rng.normal(0, 0.02)
                     new_y[i] += rng.normal(0, 0.02)
                     new_x[j] -= rng.normal(0, 0.02)
                     new_y[j] -= rng.normal(0, 0.02)
                     new_x[i] = np.clip(new_x[i], 0.01, w - 0.01)
                     new_y[i] = np.clip(new_y[i], 0.01, h - 0.01)
                     new_x[j] = np.clip(new_x[j], 0.01, w - 0.01)
                     new_y[j] = np.clip(new_y[j], 0.01, h - 0.01)

            new_energy = _energy(new_x, new_y, new_r, w, h, penalty_weight=100.0)
            
            delta_e = new_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                x, y, r = new_x, new_y, new_r
                current_energy = new_energy
                
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                # Adaptive cooling: cool faster if stuck
                if stagnation_counter > 100:
                    T *= 0.95
                    stagnation_counter = 0
                else:
                    T *= cooling_rate
            else:
                T *= cooling_rate
                
        # End of SA loop
        # Try to expand radii from the best local state found
        x_exp, y_exp, r_exp = _expand_radii(best_local_x, best_local_y, best_local_r, w, h)
        
        # Check validity and value
        if _is_valid(x_exp, y_exp, r_exp, w, h):
            val = float(np.sum(r_exp))
            if val > best_value:
                best_value = val
                # Store result
                best_result = np.column_stack((x_exp, y_exp, r_exp))
                
    # Fallback if no valid result found (should not happen)
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([cx, cy, r])
        best_result = np.array(cells)
        
    return best_result
