"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _energy(pos, radii, penalty_weight=1000.0):
    """
    Calculate the energy (cost) of the configuration.
    Lower is better. We want to minimize penalty and maximize radii.
    Energy = penalty - sum(radii)
    """
    # Boundary violations - use soft violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations using squared distances for efficiency
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq[mask] - dist_sq[mask])
    overlap_pen = np.sum(viol * viol)
    
    total_pen = boundary_pen + overlap_pen
    score = np.sum(radii)
    
    return penalty_weight * total_pen - score

def _is_valid(pos, radii, tol=1e-8):
    """
    Strict validity check.
    """
    if np.any(pos[:, 0] < -tol) or np.any(pos[:, 0] > 1.0 + tol):
        return False
    if np.any(pos[:, 1] < -tol) or np.any(pos[:, 1] > 1.0 + tol):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # dist < min_dist => dist_sq < min_dist^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _shrink_to_valid(pos, radii, tol=1e-6):
    """
    Shrink radii and adjust positions slightly to ensure validity.
    This is a safety net.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Iterative shrinking
    for _ in range(100):
        valid = True
        
        # Check bounds
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + tol:
                radii[i] = max(0.0, max_r - tol)
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - tol:
                    # Shrink both proportionally to fit
                    # dist >= r_i + r_j
                    # k * (r_i + r_j) <= dist => k <= dist / (r_i + r_j)
                    k = max(0.0, dist / (sum_r + 1e-12))
                    k = min(k, 1.0) # Only shrink
                    radii[i] = radii[i] * k
                    radii[j] = radii[j] * k
                    valid = False
        
        if valid:
            break
            
    return pos, radii

def _energy_delta_single(pos, radii, i, new_x, new_y):
    """Fast energy delta for moving circle i to (new_x, new_y)."""
    old_x, old_y = pos[i]
    r = radii[i]
    
    # Boundary change
    old_bv = np.maximum(0.0, r - old_x) + np.maximum(0.0, old_x - (1.0 - r)) + \
             np.maximum(0.0, r - old_y) + np.maximum(0.0, old_y - (1.0 - r))
    new_bv = np.maximum(0.0, r - new_x) + np.maximum(0.0, new_x - (1.0 - r)) + \
             np.maximum(0.0, r - new_y) + np.maximum(0.0, new_y - (1.0 - r))
    delta_boundary = new_bv * new_bv - old_bv * old_bv
    
    # Overlap changes with all other circles
    delta_overlap = 0.0
    for j in range(N):
        if j == i:
            continue
        dx_old = old_x - pos[j, 0]
        dy_old = old_y - pos[j, 1]
        dist_sq_old = dx_old * dx_old + dy_old * dy_old
        min_d = r + radii[j]
        viol_old = max(0.0, min_d * min_d - dist_sq_old)
        
        dx_new = new_x - pos[j, 0]
        dy_new = new_y - pos[j, 1]
        dist_sq_new = dx_new * dx_new + dy_new * dy_new
        viol_new = max(0.0, min_d * min_d - dist_sq_new)
        
        delta_overlap += viol_new * viol_new - viol_old * viol_old
    
    return delta_boundary + delta_overlap

def _energy_delta_radius(pos, radii, i, new_r):
    """Fast energy delta for changing radius of circle i."""
    old_r = radii[i]
    x, y = pos[i]
    
    # Boundary change
    old_bv = np.maximum(0.0, old_r - x) + np.maximum(0.0, x - (1.0 - old_r)) + \
             np.maximum(0.0, old_r - y) + np.maximum(0.0, y - (1.0 - old_r))
    new_bv = np.maximum(0.0, new_r - x) + np.maximum(0.0, x - (1.0 - new_r)) + \
             np.maximum(0.0, new_r - y) + np.maximum(0.0, y - (1.0 - new_r))
    delta_boundary = new_bv * new_bv - old_bv * old_bv
    
    # Overlap changes with all other circles
    delta_overlap = 0.0
    for j in range(N):
        if j == i:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist_sq = dx * dx + dy * dy
        
        min_d_old = old_r + radii[j]
        viol_old = max(0.0, min_d_old * min_d_old - dist_sq)
        
        min_d_new = new_r + radii[j]
        viol_new = max(0.0, min_d_new * min_d_new - dist_sq)
        
        delta_overlap += viol_new * viol_new - viol_old * viol_old
    
    return delta_boundary + delta_overlap

def _sa_packing(init_pos, init_radii, steps=20000, seed=0, deadline=None):
    """
    Simulated Annealing to pack circles with adaptive step sizes and multiple move types.
    Uses incremental energy evaluation for speed.
    """
    rng = np.random.RandomState(seed)
    pos = init_pos.copy().astype(float)
    radii = init_radii.copy().astype(float)
    
    current_energy = _energy(pos, radii)
    best_energy = current_energy
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    min_T = 1e-8
    
    # Adaptive step size parameters
    pos_step = 0.05
    rad_step = 0.012
    
    # Track acceptance rates for adaptive step sizing
    accept_count = 0
    total_count = 0
    last_adjust_step = 0
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Calculate cooling rate based on progress
        progress = step / max(steps, 1)
        T = max(min_T, 0.12 * (1 - progress) ** 3.0 + 0.001 * (1 - progress))
        
        # Generate a move - multiple strategies
        move_type = rng.random()
        
        if move_type < 0.45:
            # Move position of a random circle
            i = rng.randint(N)
            angle = rng.uniform(0, 2 * np.pi)
            magnitude = abs(rng.normal(0, pos_step))
            delta_x = magnitude * np.cos(angle)
            delta_y = magnitude * np.sin(angle)
            
            new_x = np.clip(pos[i, 0] + delta_x, 0.005, 0.995)
            new_y = np.clip(pos[i, 1] + delta_y, 0.005, 0.995)
            
            delta_pen = _energy_delta_single(pos, radii, i, new_x, new_y)
            delta_E = 1000.0 * delta_pen  # Using default penalty weight
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                pos[i, 0] = new_x
                pos[i, 1] = new_y
                current_energy += delta_E
                accept_count += 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    
        elif move_type < 0.75:
            # Grow or shrink radius of a random circle
            i = rng.randint(N)
            grow_prob = 0.7 if progress < 0.3 else (0.6 if progress < 0.6 else 0.5)
            if rng.random() < grow_prob:
                delta_r = abs(rng.normal(0, rad_step))
            else:
                delta_r = -abs(rng.normal(0, rad_step))
                
            new_r = np.clip(radii[i] + delta_r, 1e-4, 0.4)
            
            delta_pen = _energy_delta_radius(pos, radii, i, new_r)
            delta_E = 1000.0 * delta_pen - (new_r - radii[i])  # radius change affects score
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                radii[i] = new_r
                current_energy += delta_E
                accept_count += 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    
        elif move_type < 0.90:
            # Move two nearby circles (pairwise interaction)
            i, j = rng.choice(N, size=2, replace=False)
            if i == j:
                continue
            angle = rng.uniform(0, 2 * np.pi)
            mag = abs(rng.normal(0, pos_step * 0.6))
            dx = mag * np.cos(angle)
            dy = mag * np.sin(angle)
            
            old_xi, old_yi = pos[i, 0], pos[i, 1]
            old_xj, old_yj = pos[j, 0], pos[j, 1]
            
            new_xi = np.clip(old_xi + dx, 0.005, 0.995)
            new_yi = np.clip(old_yi + dy, 0.005, 0.995)
            new_xj = np.clip(old_xj - dx, 0.005, 0.995)
            new_yj = np.clip(old_yj - dy, 0.005, 0.995)
            
            # Compute combined energy delta
            old_e = _energy(pos, radii)
            pos[i, 0], pos[i, 1] = new_xi, new_yi
            pos[j, 0], pos[j, 1] = new_xj, new_yj
            new_e = _energy(pos, radii)
            delta_E = new_e - old_e
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                current_energy = new_e
                accept_count += 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                pos[i, 0], pos[i, 1] = old_xi, old_yi
                pos[j, 0], pos[j, 1] = old_xj, old_yj
        
        else:
            # Random kick: teleport a circle to a random position
            i = rng.randint(N)
            old_x, old_y = pos[i]
            pos[i, 0] = rng.uniform(0.05, 0.95)
            pos[i, 1] = rng.uniform(0.05, 0.95)
            
            new_energy = _energy(pos, radii)
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                current_energy = new_energy
                accept_count += 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                pos[i, 0] = old_x
                pos[i, 1] = old_y
        
        total_count += 1
        
        # Adaptive step size adjustment every 300 steps
        if step - last_adjust_step >= 300:
            accept_rate = accept_count / max(total_count - last_adjust_step, 1)
            if accept_rate < 0.12:
                pos_step *= 1.2
                rad_step *= 1.2
            elif accept_rate > 0.50:
                pos_step *= 0.85
                rad_step *= 0.85
            
            pos_step = np.clip(pos_step, 0.005, 0.15)
            rad_step = np.clip(rad_step, 0.002, 0.05)
            
            accept_count = 0
            last_adjust_step = step
        
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """
    Final high-precision polishing using L-BFGS-B with multiple penalty weights and restarts.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.45))
        
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    def _objective(x, penalty_weight=1000.0):
        pos = x[:2*N].reshape(N, 2)
        radii = np.maximum(x[2*N:], 1e-6)
        
        # Boundary violations
        bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
             np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
        boundary_pen = np.sum(bv * bv)
        
        # Overlap violations
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        min_dist_sq = min_dist * min_dist
        
        viol = np.maximum(0.0, min_dist_sq - dist_sq)
        overlap_pen = np.sum(viol[mask] * viol[mask])
        
        return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

    best_x = x.copy()
    best_obj = _objective(x, 1000.0)
    
    # Try multiple penalty weights with perturbations
    for pw in [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]:
        # Multiple perturbations per weight
        for trial in range(3):
            if time.time() > deadline:
                break
            rng_local = np.random.RandomState(int(pw) * 100 + trial)
            perturb_scale = 0.005 + 0.005 * trial
            x_perturbed = best_x + rng_local.normal(0, perturb_scale, size=len(best_x))
            x_perturbed[:2*N] = np.clip(x_perturbed[:2*N], 0, 1)
            x_perturbed[2*N:] = np.clip(x_perturbed[2*N:], 1e-4, 0.45)
            
            try:
                result = minimize(_objective, x_perturbed, method='L-BFGS-B', bounds=bounds,
                                  args=(pw,),
                                  options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14,
                                           'maxls': 100})
                obj_val = _objective(result.x, pw)
                if obj_val < best_obj - 1e-12:
                    best_obj = obj_val
                    best_x = result.x.copy()
            except Exception:
                pass
    
    # Final ultra-high precision polish
    for pw in [20000.0, 50000.0]:
        if time.time() > deadline:
            break
        try:
            result = minimize(_objective, best_x, method='L-BFGS-B', bounds=bounds,
                              args=(pw,),
                              options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-15,
                                       'maxls': 120})
            if _objective(result.x, pw) < _objective(best_x, pw) - 1e-14:
                best_x = result.x.copy()
        except Exception:
            pass
    
    pos_out = best_x[:2*N].reshape(N, 2)
    radii_out = np.maximum(best_x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    inits = []
    
    # 1. Grid Init
    pos_grid = np.zeros((N, 2))
    rad_grid = np.zeros(N)
    spacing = 1.0 / 5.0
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos_grid[idx] = [(i + 0.5) * spacing, (j + 0.5) * spacing]
                rad_grid[idx] = spacing * 0.45
    if N > 25:
        pos_grid[25] = [0.5, 0.5]
        rad_grid[25] = 0.05
    inits.append((pos_grid, rad_grid))
    
    # 2. Random Init
    rng = np.random.RandomState(42)
    pos_rand = rng.uniform(0.1, 0.9, (N, 2))
    rad_rand = rng.uniform(0.05, 0.1, N)
    inits.append((pos_rand, rad_rand))
    
    # 3. Hexagonal Init
    pos_hex = np.zeros((N, 2))
    rad_hex = np.zeros(N)
    r = 0.11
    dx = r * np.sqrt(3) * 1.05
    dy = r * 1.5 * 1.05
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = r + offset + col * dx
            if 0 <= x <= 1 and 0 <= y <= 1:
                pos_hex[k] = [x, y]
                rad_hex[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        pos_hex[k] = [0.5, 0.5]
        rad_hex[k] = 0.01
        k += 1
    inits.append((pos_hex, rad_hex))
    
    # 4. Corner-focused Init (often good for packing)
    pos_corner = np.zeros((N, 2))
    rad_corner = np.zeros(N)
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for i, c in enumerate(corners):
        pos_corner[i] = c
        rad_corner[i] = 0.12
    # Fill rest with edge-focused positions
    edges = [
        [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.5, 0.5]
    ]
    for i in range(4, N):
        if i - 4 < len(edges):
            pos_corner[i] = edges[i - 4]
            rad_corner[i] = 0.06
        else:
            pos_corner[i] = rng.uniform(0.2, 0.8, 2)
            rad_corner[i] = 0.05
    inits.append((pos_corner, rad_corner))
    
    # 5. Dense corner cluster init
    pos_dense = np.zeros((N, 2))
    rad_dense = np.zeros(N)
    # Place 4 large circles in corners
    corner_pos = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for i, cp in enumerate(corner_pos):
        pos_dense[i] = cp
        rad_dense[i] = 0.14
    # Place remaining circles in between
    mid_positions = [
        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
        [0.5, 0.5]
    ]
    for i in range(4, N):
        if i - 4 < len(mid_positions):
            pos_dense[i] = mid_positions[i - 4]
            rad_dense[i] = 0.04
        else:
            pos_dense[i] = rng.uniform(0.2, 0.8, 2)
            rad_dense[i] = 0.03
    inits.append((pos_dense, rad_dense))
    
    # 6. Symmetric diagonal init
    pos_diag = np.zeros((N, 2))
    rad_diag = np.zeros(N)
    # Place circles along diagonals and edges
    diag_positions = [
        [0.07, 0.07], [0.93, 0.93], [0.93, 0.07], [0.07, 0.93],
        [0.5, 0.07], [0.5, 0.93], [0.07, 0.5], [0.93, 0.5],
        [0.25, 0.25], [0.75, 0.75], [0.75, 0.25], [0.25, 0.75],
        [0.5, 0.5]
    ]
    for i, dp in enumerate(diag_positions):
        if i < N:
            pos_diag[i] = dp
            rad_diag[i] = 0.08 if i < 4 else 0.05
    inits.append((pos_diag, rad_diag))
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        # Run SA
        time_left = deadline - time.time()
        # Use more steps for better optimization with faster energy eval
        steps = int(min(60000, time_left * 1200))
        
        pos_sa, rad_sa = _sa_packing(pos_init, rad_init, steps=steps, seed=idx*100 + 42, deadline=deadline - 12)
        
        # Polish
        if time.time() < deadline - 5:
            pos_pol, rad_pol = _polish(pos_sa, rad_sa, deadline)
            
            # Validate and shrink
            pos_val, rad_val = _shrink_to_valid(pos_pol, rad_pol)
            
            if _is_valid(pos_val, rad_val):
                score = np.sum(rad_val)
                if score > best_score:
                    best_score = score
                    best_pos = pos_val.copy()
                    best_radii = rad_val.copy()
                    
                    # Try to further improve with additional SA from this good solution
                    if time.time() < deadline - 18:
                        pos_extra, rad_extra = _sa_packing(pos_val, rad_val, steps=8000, 
                                                           seed=idx*1000 + 777, deadline=deadline - 10)
                        if time.time() < deadline - 5:
                            pos_pol2, rad_pol2 = _polish(pos_extra, rad_extra, deadline)
                            pos_val2, rad_val2 = _shrink_to_valid(pos_pol2, rad_pol2)
                            if _is_valid(pos_val2, rad_val2):
                                score2 = np.sum(rad_val2)
                                if score2 > best_score:
                                    best_score = score2
                                    best_pos = pos_val2.copy()
                                    best_radii = rad_val2.copy()
                    
    # If no valid solution found, fallback to a simple grid
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                if i*5+j < N-1:
                    pos[i*5+j] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[i*5+j] = 0.09
        pos[N-1] = [0.5, 0.02]
        radii[N-1] = 0.01
        best_pos, best_radii = _shrink_to_valid(pos, radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
