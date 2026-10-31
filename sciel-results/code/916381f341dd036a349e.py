"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

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
    
    # Use a smoother penalty: (violation)^2 / (min_dist^2 + epsilon)
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom
    
    overlap_penalty = np.sum(smooth_overlap[mask])
    
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
    
    for _ in range(100):
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
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 2000 random points for better coverage
        for _ in range(2000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.005:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.005:
            best_x, best_y = 0.5, 0.5
            best_r = 0.005
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98
        
    return pos, radii

def _initial_lattice():
    """
    Generate an initial packing based on a lattice structure.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    points = []
    for i in range(5):
        for j in range(5):
            points.append([0.1 + i*0.2, 0.1 + j*0.2])
            
    for i in range(min(N, len(points))):
        pos[i] = points[i]
        radii[i] = 0.095
        
    if N > len(points):
        for i in range(len(points), N):
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            rng = np.random.RandomState(i * 777)
            for _ in range(500):
                x = rng.uniform(0.1, 0.9)
                y = rng.uniform(0.1, 0.9)
                max_r_wall = min(x, 1.0-x, y, 1.0-y)
                max_r_circ = max_r_wall
                for j in range(i):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                if max_r_circ > best_r:
                    best_r = max_r_circ
                    best_x = x
                    best_y = y
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.98
            
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_anneal(pos, radii, time_limit=5.0):
    """
    Force-directed annealing to escape local minima.
    Moves circles apart when overlapping, and grows radii when possible.
    """
    start = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    T = 0.05  # Initial temperature
    cooling = 0.98
    
    while time.time() - start < time_limit:
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            if x < r: forces[i, 0] += (r - x) * 10
            if x > 1-r: forces[i, 0] -= (x - (1-r)) * 10
            if y < r: forces[i, 1] += (r - y) * 10
            if y > 1-r: forces[i, 1] -= (y - (1-r)) * 10
            
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Repulsive force
                    f = (sum_r - dist) * 5
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Apply forces with noise
        for i in range(N):
            noise = np.random.normal(0, T, 2)
            pos[i] += forces[i] * 0.01 + noise * 0.01
            
            # Keep in bounds
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
            
            # Try to grow radius
            max_r = min(pos[i, 0], 1-pos[i, 0], pos[i, 1], 1-pos[i, 1])
            for j in range(N):
                if i != j:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, dist - radii[j])
            
            if max_r > radii[i]:
                radii[i] = max_r * 0.99
            else:
                radii[i] = max(max_r * 0.99, 1e-5)
        
        T *= cooling
    
    return pos, radii

def _max_radii_for_positions(pos, radii):
    """Vectorized: compute max possible radius for each circle given positions."""
    # Wall constraints
    max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                       np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    # Circle constraints: for each i, max over j != i of (dist_ij - r_j)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    # dist[i,j] - radii[j], exclude diagonal
    np.fill_diagonal(dist, np.inf)
    circle_max = dist - radii[np.newaxis, :]
    max_r = np.minimum(max_r, np.min(circle_max, axis=1))
    return max_r

def _contact_graph_relax(pos, radii, time_limit=10.0):
    """
    Vectorized greedy growth + local move strategy.
    """
    start = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(2000):
        if time.time() - start > time_limit:
            break
        
        # 1. Grow radii (vectorized)
        max_r = _max_radii_for_positions(pos, radii)
        grow_mask = max_r > radii * 1.001
        changed = np.any(grow_mask)
        if changed:
            radii[grow_mask] = max_r[grow_mask] * 0.999
        
        if not changed:
            # Try to move circles to create space
            any_moved = False
            for i in range(N):
                best_gain = 0
                best_dx, best_dy = 0, 0
                current_r = radii[i]
                
                # Try 8 directions
                for dx in [-0.005, -0.01, 0.01, 0.02]:
                    for dy in [-0.005, -0.01, 0.01, 0.02]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = pos[i, 0] + dx
                        ny = pos[i, 1] + dy
                        if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                            continue
                        
                        # Compute max radius at new position (vectorized over j)
                        mr = min(nx, 1.0 - nx, ny, 1.0 - ny)
                        ddx = nx - pos[:, 0]
                        ddy = ny - pos[:, 1]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        d[i] = np.inf
                        mr = min(mr, np.min(d - radii))
                        
                        if mr > current_r + 1e-4:
                            gain = mr - current_r
                            if gain > best_gain:
                                best_gain = gain
                                best_dx, best_dy = dx, dy
                
                if best_dx != 0 or best_dy != 0:
                    pos[i, 0] += best_dx
                    pos[i, 1] += best_dy
                    # Recompute max radius at new position
                    mr = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                    ddx = pos[i, 0] - pos[:, 0]
                    ddy = pos[i, 1] - pos[:, 1]
                    d = np.sqrt(ddx * ddx + ddy * ddy)
                    d[i] = np.inf
                    mr = min(mr, np.min(d - radii))
                    radii[i] = mr * 0.999
                    any_moved = True
            
            if not any_moved:
                # If no progress, add some noise to escape local minima
                pos += np.random.normal(0, 0.003, (N, 2))
                pos = np.clip(pos, 0.01, 0.99)
    
    # Final grow
    max_r = _max_radii_for_positions(pos, radii)
    radii = np.maximum(max_r * 0.999, 1e-5)
    
    return pos, radii

def _boundary_constraints(x):
    """Equality/inequality constraints for SLSQP: boundary."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    # x >= r, 1-x >= r, y >= r, 1-y >= r
    return [
        pos[:, 0] - radii,
        (1.0 - pos[:, 0]) - radii,
        pos[:, 1] - radii,
        (1.0 - pos[:, 1]) - radii,
    ]

def _overlap_constraints(x):
    """Inequality constraints for SLSQP: no overlap. dist >= ri + rj."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    # dist - min_dist >= 0 for all pairs
    return [dist[mask] - min_dist[mask]]

def _slsqp_optimize(x0, time_limit=20.0):
    """Run SLSQP with hard constraints."""
    start = time.time()
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    # Pairwise overlap constraints (only close pairs to reduce constraint count)
    pos0 = x0[:2*N].reshape(N, 2)
    r0 = x0[2*N:]
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((pos0[i,0]-pos0[j,0])**2 + (pos0[i,1]-pos0[j,1])**2)
            if d < (r0[i] + r0[j]) * 1.5:
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                    np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - x[2*N+i] - x[2*N+j]})
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    try:
        result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12})
        if time.time() - start < time_limit and result.x is not None:
            return result.x
    except Exception:
        pass
    return x0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Strategy: Multi-start with different initializations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Random initializations
    for i in range(3):
        pos_p, rad_p = _initial_random(100 + i)
        inits.append((pos_p, rad_p))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    # Multi-stage optimization with increasing penalty weight
    penalty_stages = [100.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: Coarse optimization with low penalty
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(penalty_stages[0],),
                              options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
            x0 = result.x
        except Exception:
            pass
        
        # Stage 2: Medium penalty
        if time.time() < deadline - 35:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: High penalty for final polish
        if time.time() < deadline - 40:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
                x0 = result.x
            except Exception:
                pass
            
            x_opt = x0
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # SLSQP with hard constraints (measured to improve score)
        if time.time() < deadline - 45 and idx < 3:
            x_slsqp = _slsqp_optimize(x0, time_limit=15.0)
            if x_slsqp is not None:
                pos_s = x_slsqp[:2*N].reshape(N, 2)
                radii_s = np.maximum(x_slsqp[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_s, radii_s)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Contact graph relaxation on the best result
        if time.time() < deadline - 30:
            pos_cg, radii_cg = _contact_graph_relax(best_pos, best_radii, time_limit=5.0)
            pos_cg, radii_cg = _post_process(pos_cg, radii_cg)
            score_cg = np.sum(radii_cg)
            if _is_valid(pos_cg, radii_cg) and score_cg > best_score:
                best_score = score_cg
                best_pos = pos_cg.copy()
                best_radii = radii_cg.copy()

    # Final high-precision polish on the best result
    if time.time() < deadline - 15:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
        
        # Final SLSQP polish
        if time.time() < deadline - 10:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            x_slsqp = _slsqp_optimize(x0, time_limit=10.0)
            if x_slsqp is not None:
                pos_s = x_slsqp[:2*N].reshape(N, 2)
                radii_s = np.maximum(x_slsqp[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_s, radii_s)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()

    # Final contact graph relaxation
    if time.time() < deadline - 5:
        pos_cg, radii_cg = _contact_graph_relax(best_pos, best_radii, time_limit=3.0)
        pos_cg, radii_cg = _post_process(pos_cg, radii_cg)
        score_cg = np.sum(radii_cg)
        if _is_valid(pos_cg, radii_cg) and score_cg > best_score:
            best_score = score_cg
            best_pos = pos_cg.copy()
            best_radii = radii_cg.copy()

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
