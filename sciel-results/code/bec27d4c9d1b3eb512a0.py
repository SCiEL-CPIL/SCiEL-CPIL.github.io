"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softmax_penalty(violations, beta=50.0):
    """
    Smooth approximation of sum(max(0, v)^2) using log-sum-exp.
    For v > 0: ~ v^2; for v < 0: ~ 0 (exponentially small).
    """
    v = np.asarray(violations)
    clipped = np.clip(v, -10.0, 10.0)
    sp = np.log1p(np.exp(beta * clipped)) / beta
    return np.sum(clipped * sp)

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (Smooth Squared Penalty)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap Violations (Vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Smoother penalty: (violation)^2 / (min_dist^2 + epsilon)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _objective_smooth(x, penalty_weight=500.0, beta=30.0):
    """
    Objective with log-sum-exp style smooth penalties for better gradient flow.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    radii = np.maximum(radii, 1e-6)
    
    # Boundary violations (raw, can be negative)
    left_viol = radii - pos[:, 0]
    right_viol = pos[:, 0] - (1.0 - radii)
    bottom_viol = radii - pos[:, 1]
    top_viol = pos[:, 1] - (1.0 - radii)
    
    boundary_penalty = (_softmax_penalty(left_viol, beta) + _softmax_penalty(right_viol, beta) +
                        _softmax_penalty(bottom_viol, beta) + _softmax_penalty(top_viol, beta))
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # violation = min_dist_sq - dist_sq (can be negative when separated)
    violation_raw = min_dist_sq - dist_sq
    violation_masked = violation_raw[mask]
    overlap_penalty = _softmax_penalty(violation_masked, beta)
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii to make them just touch
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    This creates a valid, dense configuration that serves as a strong starting point.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        # Try to find a position for the next circle
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 5000 random points for better coverage
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_corner_biased(seed=7):
    """
    Initialize with circles biased toward corners and edges,
    which often yields better packings for maximizing sum of radii.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place 4 circles near corners
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
    # Place remaining circles with preference for edge regions
    for i in range(4, N):
        # Choose a random edge or interior point
        if rng.random() < 0.4:
            # Edge-biased
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:  # bottom
                x, y = t, 0.08
            elif edge == 1:  # top
                x, y = t, 0.92
            elif edge == 2:  # left
                x, y = 0.08, t
            else:  # right
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        # Find max radius at this position
        max_r = min(x, 1.0-x, y, 1.0-y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            # Try to find a better spot
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0-x2, y2, 1.0-y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
    
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    # Force validity by shrinking
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    This often provides a better starting point for dense packings than random.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            # Center the last circle
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        # Offset odd rows
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            # Check if it fits in the box
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    # If we didn't fill all 26, fill remaining with small circles in corners
    while k < N:
        # Find a spot
        placed = False
        for _ in range(100):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            # Force place tiny
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _force_directed_anneal(pos, radii, n_steps=5000, t0=0.05, t_end=0.001, seed=0):
    """
    Force-directed annealing: treat circles as particles with repulsive forces
    when overlapping and attractive forces to stay in bounds.
    Gradually cool to find a dense packing.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(n_steps):
        # Temperature schedule (exponential cooling)
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces: push inward if circle is too close to wall
        for i in range(n):
            r = radii[i]
            # Left wall
            if pos[i, 0] < r:
                forces[i, 0] += (r - pos[i, 0]) * 10.0
            # Right wall
            if pos[i, 0] > 1.0 - r:
                forces[i, 0] -= (pos[i, 0] - (1.0 - r)) * 10.0
            # Bottom wall
            if pos[i, 1] < r:
                forces[i, 1] += (r - pos[i, 1]) * 10.0
            # Top wall
            if pos[i, 1] > 1.0 - r:
                forces[i, 1] -= (pos[i, 1] - (1.0 - r)) * 10.0
        
        # Pairwise repulsive forces when overlapping
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap * 5.0
                    fy = (dy / dist) * overlap * 5.0
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Apply forces with thermal noise
        for i in range(n):
            pos[i, 0] += forces[i, 0] * 0.01 + rng.normal(0, temp * 0.01)
            pos[i, 1] += forces[i, 1] * 0.01 + rng.normal(0, temp * 0.01)
            
            # Clamp to valid region
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r * 0.5, 1.0 - r * 0.5)
            pos[i, 1] = np.clip(pos[i, 1], r * 0.5, 1.0 - r * 0.5)
        
        # Every 100 steps, try to grow radii slightly
        if step % 100 == 50:
            for i in range(n):
                # Check if we can grow
                min_allowed = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
                for j in range(n):
                    if j != i:
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_allowed = min(min_allowed, dist - radii[j])
                if min_allowed > radii[i]:
                    radii[i] = min(min_allowed * 0.98, radii[i] * 1.02 + 0.001)
        
        # Check validity and track best
        if step % 200 == 0:
            valid = True
            # Quick boundary check
            for i in range(n):
                r = radii[i]
                if pos[i,0] - r < -1e-6 or pos[i,0] + r > 1.0 + 1e-6 or \
                   pos[i,1] - r < -1e-6 or pos[i,1] + r > 1.0 + 1e-6:
                    valid = False
                    break
            if valid:
                # Quick overlap check
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        dist_sq = dx*dx + dy*dy
                        min_d = radii[i] + radii[j]
                        if dist_sq < (min_d - 1e-6)**2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """
    High-precision polishing stage using SLSQP with tight tolerances.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    # SLSQP polish
    try:
        result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                          args=(2000.0,),
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _objective(x, 2000.0) - 1e-12:
            x = result.x
    except Exception:
        pass
    
    # L-BFGS-B polish
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x, 5000.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Strategy: Multi-start with different initializations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Corner-biased
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    # 4. Random (2 variations)
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    # Multi-stage optimization with both standard and smooth objectives
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective with increasing penalties
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            # Check validity and update best
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: Smooth objective for better gradient flow
        if time.time() < deadline - 10:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 3: Force-directed annealing from current state
        if time.time() < deadline - 8:
            try:
                pos_fd, rad_fd = _force_directed_anneal(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=3000, t0=0.03, t_end=0.0005,
                    seed=idx * 100
                )
                pos_pp, radii_pp = _post_process(pos_fd, rad_fd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final high-precision polish: SLSQP with very tight tolerances
    if time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback if invalid
    if not _is_valid(best_pos, best_radii):
         pos = np.zeros((N, 2))
         radii = np.zeros(N)
         k = 0
         for i in range(5):
             for j in range(5):
                 if k < N-1:
                     pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                     radii[k] = 0.095
                     k += 1
         pos[k] = [0.5, 0.02]
         radii[k] = 0.01
         best_pos, best_radii = pos, radii
         best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
