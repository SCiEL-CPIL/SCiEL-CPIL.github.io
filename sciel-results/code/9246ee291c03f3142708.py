import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_violations_and_score(pos, radii, w, h):
    """
    Compute constraint violations and the sum of radii.
    Returns (violation_sum, sum_radii, valid_flag)
    """
    n = len(radii)
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    # Wall violations: circle must be inside [0, w] x [0, h]
    # Violation is max(0, r - dist_to_wall)
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    # Circle-circle violations
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    # Overlap if dist < r_i + r_j
    # Violation is max(0, r_i + r_j - dist)
    overlap_sq = np.zeros_like(d_sq_upper)
    overlap_mask = d_sq_upper < r_sum_upper**2
    if np.any(overlap_mask):
        d_overlap = np.sqrt(d_sq_upper[overlap_mask])
        overlap_val = r_sum_upper[overlap_mask] - d_overlap
        overlap_sq[overlap_mask] = overlap_val**2
        
    circle_viol = np.sum(overlap_sq)
    
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    valid = total_viol < 1e-8
    return total_viol, sum_rad, valid

def _objective(x, w, h, penalty_weight=100.0):
    """
    Objective function for optimization.
    x = [x1, y1, r1, x2, y2, r2, ...]
    We minimize: penalty_weight * (Sum of Squared Violations) - (Sum of Radii)
    """
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    # Clip radii to be positive for stability during optimization
    radii = np.maximum(radii, 1e-6)
    
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    wall_viol = np.sum(np.maximum(0, radii - d_left)**2) + \
                np.sum(np.maximum(0, radii - d_right)**2) + \
                np.sum(np.maximum(0, radii - d_bottom)**2) + \
                np.sum(np.maximum(0, radii - d_top)**2)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    overlap_sq = np.zeros_like(d_sq_upper)
    overlap_mask = d_sq_upper < r_sum_upper**2
    if np.any(overlap_mask):
        d_overlap = np.sqrt(d_sq_upper[overlap_mask])
        overlap_val = r_sum_upper[overlap_mask] - d_overlap
        overlap_sq[overlap_mask] = overlap_val**2
        
    circle_viol = np.sum(overlap_sq)
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    return penalty_weight * total_viol - sum_rad

def _hexagonal_start(w, h, num_circles):
    """
    Generate a hexagonal lattice packing as initial configuration.
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
    # For 21 circles, a 5x5 grid with hexagonal offset is a good start
    # But we need to fit it in w x h.
    # Let's assume a square aspect ratio for initialization, then scale.
    # Or just place them in a grid.
    
    cols = 5
    rows = 5
    
    # Calculate ideal spacing for a square packing first
    # If we pack 25 circles, we have 4 extra spots? No, 21 circles.
    # Let's just use a uniform grid of 5x5 and remove 4? 
    # Or better: 3 rows of 7? 7*3=21.
    # Let's try 3 rows of 7.
    cols = 7
    rows = 3
    
    for i in range(num_circles):
        row = i // cols
        col = i % cols
        
        x = (col + 0.5) / cols * w
        y = (row + 0.5) / rows * h
        
        # Hexagonal offset for odd rows
        if row % 2 == 1:
            x += w / (2 * cols)
        
        pos[i, 0] = x
        pos[i, 1] = y
    
    # Compute initial radii based on minimum distance
    min_dist = np.inf
    for i in range(num_circles):
        for j in range(i+1, num_circles):
            d = np.sqrt(np.sum((pos[i] - pos[j])**2))
            min_dist = min(min_dist, d)
    
    for i in range(num_circles):
        d_wall = min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
        min_dist = min(min_dist, d_wall)
    
    radii[:] = 0.5 * min_dist # Start slightly smaller to avoid initial overlap
    return pos, radii

def _random_start(w, h, num_circles, seed=None):
    """
    Generate a random initial configuration.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    pos = rng.rand(num_circles, 2) * np.array([w, h])
    max_r = min(w, h) / 2.0
    radii = rng.rand(num_circles) * 0.1 * max_r
    
    return pos, radii

def _md_step(pos, radii, w, h, temp, dt):
    """
    Perform one step of molecular dynamics / gradient descent.
    Returns updated pos, radii, and violation.
    """
    n = len(radii)
    force_pos = np.zeros_like(pos)
    force_rad = np.zeros_like(radii)
    
    # 1. Wall forces
    # Push away from walls if too close
    d_left = pos[:, 0]
    d_right = w - pos[:, 0]
    d_bottom = pos[:, 1]
    d_top = h - pos[:, 1]
    
    # If d < r, push out
    # Force is proportional to penetration
    pen_left = np.maximum(0, radii - d_left)
    pen_right = np.maximum(0, radii - d_right)
    pen_bottom = np.maximum(0, radii - d_bottom)
    pen_top = np.maximum(0, radii - d_top)
    
    force_pos[:, 0] += pen_left - pen_right
    force_pos[:, 1] += pen_bottom - pen_top
    
    # 2. Circle-circle forces
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq_upper = dist_sq[mask]
    r_sum_upper = r_sum[mask]
    
    # Only calculate force for overlapping or close circles
    close_mask = d_sq_upper < (r_sum_upper * 1.1)**2 # Slightly larger than overlap
    
    if np.any(close_mask):
        d_overlap = np.sqrt(np.maximum(d_sq_upper[close_mask], 1e-12))
        overlap_val = r_sum_upper[close_mask] - d_overlap
        # Force is repulsive if overlap > 0
        # We want to push them apart
        # Gradient of dist w.r.t pos_i is (pos_i - pos_j) / dist
        # So force on i is (pos_i - pos_j) / dist * overlap_val
        
        # We need to map indices back to the full matrix
        i_idx, j_idx = np.where(mask)
        close_i = i_idx[close_mask]
        close_j = j_idx[close_mask]
        
        # Vectors from j to i
        vecs = pos[close_i] - pos[close_j]
        # Normalize
        norms = d_overlap
        # Avoid division by zero
        norms = np.maximum(norms, 1e-9)
        unit_vecs = vecs / norms[:, np.newaxis]
        
        # Force magnitude is overlap_val
        # If overlap_val > 0, push apart (force on i is +unit_vec * overlap)
        # If overlap_val < 0, no force (or attractive if we wanted, but we don't)
        forces = unit_vecs * np.maximum(overlap_val, 0)[:, np.newaxis]
        
        # Accumulate forces
        # Force on i is +f, on j is -f
        for k in range(len(close_i)):
            force_pos[close_i[k]] += forces[k]
            force_pos[close_j[k]] -= forces[k]
        
        # Force on radii: we want to reduce radii if they are overlapping?
        # Actually, the penalty method handles this. 
        # But in MD, we can add a force to reduce radii if overlap is high.
        # d(overlap)/dr_i = 1. So force on r_i is -overlap_val.
        rad_force_i = np.maximum(overlap_val, 0)
        rad_force_j = np.maximum(overlap_val, 0)
        
        np.add.at(force_rad, close_i, -rad_force_i)
        np.add.at(force_rad, close_j, -rad_force_j)

    # 3. Growth force on radii
    # We want to maximize sum of radii, so add a constant positive force to radii
    # But balance with violation forces.
    growth_force = 0.1
    force_rad += growth_force
    
    # Update positions and radii
    # Simple Euler step with damping
    pos += dt * force_pos
    radii += dt * force_rad
    
    # Clip positions to [0, w] x [0, h] roughly to avoid flying out
    pos[:, 0] = np.clip(pos[:, 0], -1, w+1)
    pos[:, 1] = np.clip(pos[:, 1], -1, h+1)
    radii = np.clip(radii, 1e-6, min(w, h))
    
    return pos, radii

def _pack_for_aspect(w, h, start_time, deadline):
    """
    Pack circles in w x h rectangle.
    Returns best (pos, radii) and score found for this aspect.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-6, max_r))
        
    # 1. Hexagonal Initialization
    pos_hex, rad_hex = _hexagonal_start(w, h, N)
    x0_hex = np.concatenate([pos_hex.flatten(), rad_hex])
    
    viol, score, valid = _compute_violations_and_score(pos_hex, rad_hex, w, h)
    if valid and score > best_score:
        best_score = score
        best_pos = pos_hex.copy()
        best_radii = rad_hex.copy()
        
    # 2. Random Initialization
    pos_rand, rad_rand = _random_start(w, h, N, seed=42)
    x0_rand = np.concatenate([pos_rand.flatten(), rad_rand])
    
    # 3. MD Phase
    # We run MD on the hexagonal start and random start
    initializations = [
        (pos_hex.copy(), rad_hex.copy()),
        (pos_rand.copy(), rad_rand.copy())
    ]
    
    for pos, radii in initializations:
        if time.time() > deadline - 10:
            break
            
        # Run MD steps
        # Start with high "temperature" (large step) and cool down
        dt = 0.05
        for step in range(200):
            if time.time() > deadline - 5:
                break
                
            pos, radii = _md_step(pos, radii, w, h, temp=1.0, dt=dt)
            
            # Cool down
            dt *= 0.99
            
            viol, score, valid = _compute_violations_and_score(pos, radii, w, h)
            
            if valid and score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
    # 4. Polish Phase with L-BFGS-B
    # Use the best result from MD as the starting point for L-BFGS
    if best_pos is not None:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            # Use a moderate penalty weight
            result = minimize(_objective, x0, args=(w, h, 50.0), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'maxls': 50})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            # Ensure radii are positive
            rad_opt = np.maximum(rad_opt, 1e-6)
            
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            if not valid:
                # Try to fix by scaling down radii
                if viol > 1e-4:
                    scale = 0.95
                    while viol > 1e-8 and scale > 0.1:
                        rad_new = rad_opt * scale
                        viol, s, valid = _compute_violations_and_score(pos_opt, rad_new, w, h)
                        if valid:
                            rad_opt = rad_new
                            score = s
                            break
                        scale *= 0.9
                    if not valid:
                        # Keep previous best
                        pass
                    else:
                        if score > best_score:
                            best_score = score
                            best_pos = pos_opt.copy()
                            best_radii = rad_opt.copy()
            else:
                score = np.sum(rad_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
                    best_radii = rad_opt.copy()
                    
        except Exception:
            pass

    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    best_global_score = 0
    best_global_result = None
    
    np.random.seed(42)
    
    # Try a few aspect ratios
    # Perimeter is 4, so w + h = 2.
    # Square is w=1, h=1.
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
            if valid:
                best_global_result = np.column_stack([pos, radii])
            elif viol < 1e-3:
                scale = 0.99
                while viol > 1e-8 and scale > 0.1:
                    radii_new = radii * scale
                    viol, s, valid = _compute_violations_and_score(pos, radii_new, w, h)
                    if valid:
                        radii = radii_new
                        break
                    scale *= 0.95
                if valid:
                    best_global_result = np.column_stack([pos, radii])
    
    # Fallback
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val * 0.9
        best_global_result = np.column_stack([pos, rad])
        
    return best_global_result
