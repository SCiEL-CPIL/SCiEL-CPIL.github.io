"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

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

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        placed = False
        rng = np.random.RandomState(k)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
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
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
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

def _force_directed_anneal(pos_init, radii_init, time_limit=10.0, seed=42):
    """
    Force-directed annealing to maximize sum of radii.
    Uses a potential energy function with repulsion for overlaps and attraction to expand.
    """
    rng = np.random.RandomState(seed)
    pos = pos_init.copy()
    radii = radii_init.copy()
    
    # Start with a slightly smaller radius to allow movement
    # radii = radii * 0.9
    
    T = 1.0  # Temperature
    T_min = 0.01
    cooling = 0.999
    step_size = 0.05
    
    start_time = time.time()
    best_sum_r = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        
        # Calculate forces
        forces = np.zeros((N, 2))
        
        # 1. Pairwise repulsion (overlap resolution)
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r:
                    # Overlap
                    # Force magnitude proportional to overlap squared?
                    # Let's use a simple linear repulsion for stability
                    overlap = sum_r - dist
                    if dist < 1e-6:
                        # Random direction
                        angle = rng.uniform(0, 2*np.pi)
                        fx = np.cos(angle)
                        fy = np.sin(angle)
                    else:
                        fx = dx / dist
                        fy = dy / dist
                    
                    f_mag = 10.0 * overlap
                    forces[i] += np.array([fx, fy]) * f_mag
                    forces[j] -= np.array([fx, fy]) * f_mag
        
        # 2. Boundary repulsion
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Left
            if x < r:
                forces[i, 0] += 10.0 * (r - x)
            # Right
            if x > 1.0 - r:
                forces[i, 0] -= 10.0 * (x - (1.0 - r))
            # Bottom
            if y < r:
                forces[i, 1] += 10.0 * (r - y)
            # Top
            if y > 1.0 - r:
                forces[i, 1] -= 10.0 * (y - (1.0 - r))
        
        # 3. "Growth" force: Encourage larger radii
        # We can simulate this by adding a small random perturbation to radii
        # and accepting if it doesn't increase overlap penalty too much.
        # Alternatively, we can just try to increase all radii slightly in each step
        # and let the repulsion forces push positions to accommodate.
        
        # Let's try increasing radii slightly
        growth_factor = 1.0 + 0.001
        new_radii = radii * growth_factor
        
        # Check if this growth causes massive overlap. If so, skip growth for this step.
        # Actually, simpler: Just update positions based on forces, and then try to grow.
        
        # Update positions
        # Add noise
        noise = rng.normal(0, T * step_size, (N, 2))
        new_pos = pos + forces * 0.01 + noise
        
        # Clamp to [0, 1] roughly to prevent flying away
        # new_pos = np.clip(new_pos, -0.5, 1.5)
        
        # Calculate "Energy" (Penalty for overlap + boundary violation)
        def calc_energy(p, r):
            e = 0.0
            for i in range(N):
                if p[i, 0] < r[i] or p[i, 0] > 1.0 - r[i]:
                    e += 100.0 * (max(0, r[i] - p[i, 0]) + max(0, p[i, 0] - (1.0 - r[i])))**2
                if p[i, 1] < r[i] or p[i, 1] > 1.0 - r[i]:
                    e += 100.0 * (max(0, r[i] - p[i, 1]) + max(0, p[i, 1] - (1.0 - r[i])))**2
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = p[i, 0] - p[j, 0]
                    dy = p[i, 1] - p[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    sum_r = r[i] + r[j]
                    if dist < sum_r:
                        e += 100.0 * (sum_r - dist)**2
            return e
        
        E_old = calc_energy(pos, radii)
        E_new_pos = calc_energy(new_pos, radii)
        
        # Accept new position if energy is lower, or with probability exp(-dE/T)
        if E_new_pos < E_old or rng.uniform() < np.exp(-(E_new_pos - E_old) / T):
            pos = new_pos
        
        # Try to grow radii
        # Only grow if current state is "good" (low energy)
        if E_old < 1.0: # Threshold for "validity"
            new_radii = radii * (1.0 + 0.0005)
            E_new_rad = calc_energy(pos, new_radii)
            
            # If growing doesn't increase energy too much, accept
            if E_new_rad < E_old + T * 0.1:
                radii = new_radii
                # Update best if sum of radii increases
                if np.sum(radii) > best_sum_r:
                    best_sum_r = np.sum(radii)
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        
        # Cool down
        T *= cooling
        
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initialize with lattice and best fit
    pos_lat, rad_lat = _initial_lattice()
    pos_bf, rad_bf = _initial_best_fit()
    
    best_pos = pos_lat.copy()
    best_radii = rad_lat.copy()
    best_score = np.sum(best_radii)
    
    # Run Force Directed Annealing on multiple starts
    # We have ~145 seconds. Let's run ~10 anneals of ~10s each?
    # Or fewer, longer anneals?
    # Let's try 5 anneals of 20s each.
    
    num_anneals = 5
    time_per_anneal = 20.0
    
    for k in range(num_anneals):
        if time.time() > deadline - 10:
            break
            
        # Alternate between lattice and best fit starts, with perturbations
        if k % 2 == 0:
            p_init, r_init = pos_lat, rad_lat
        else:
            p_init, r_init = pos_bf, rad_bf
            
        # Add random perturbation to escape local minima
        rng = np.random.RandomState(1000 + k)
        p_init = p_init + rng.uniform(-0.02, 0.02, (N, 2))
        r_init = r_init * rng.uniform(0.95, 1.05, N)
        
        # Ensure initial state is not wildly invalid (post-process)
        p_init, r_init = _post_process(p_init, r_init)
        
        # Run anneal
        p_out, r_out = _force_directed_anneal(p_init, r_init, time_limit=time_per_anneal, seed=1000+k)
        
        # Post-process result
        p_out, r_out = _post_process(p_out, r_out)
        
        if _is_valid(p_out, r_out):
            score = np.sum(r_out)
            if score > best_score:
                best_score = score
                best_pos = p_out.copy()
                best_radii = r_out.copy()
        
        # Optional: Local polish with L-BFGS on the best found so far
        # This can refine the solution significantly.
        if time.time() < deadline - 5:
            # Use the best solution found so far as start for L-BFGS
            # We need a smooth objective for L-BFGS
            # Let's use a simple penalty objective
            
            def _obj(x):
                pos = x[:2*N].reshape(N, 2)
                radii = x[2*N:]
                radii = np.maximum(radii, 1e-6)
                
                # Boundary
                left_viol = np.maximum(0.0, radii - pos[:, 0])
                right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
                bottom_viol = np.maximum(0.0, radii - pos[:, 1])
                top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
                boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                                   np.sum(bottom_viol**2) + np.sum(top_viol**2)
                
                # Overlap
                dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
                dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
                min_dist_sq = min_dist * min_dist
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                violation = np.maximum(0.0, min_dist_sq - dist_sq)
                denom = min_dist_sq + 1e-8
                overlap_penalty = np.sum((violation ** 2) / denom[mask])
                
                return -np.sum(radii) + 500.0 * (boundary_penalty + overlap_penalty)
            
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            bounds = []
            for i in range(N):
                bounds.append((0.01, 0.99))
                bounds.append((0.01, 0.99))
            for i in range(N):
                bounds.append((0.01, 0.5))
                
            try:
                res = minimize(_obj, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-10})
                if res.success:
                    pos_opt = res.x[:2*N].reshape(N, 2)
                    radii_opt = np.maximum(res.x[2*N:], 1e-4)
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    if _is_valid(pos_pp, radii_pp):
                        score = np.sum(radii_pp)
                        if score > best_score:
                            best_score = score
                            best_pos = pos_pp.copy()
                            best_radii = radii_pp.copy()
            except Exception:
                pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback
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
