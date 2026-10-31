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
    Uses asymmetric scaling: shrink the larger circle more to preserve sum.
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
        
        # 2. Overlap fix - asymmetric: shrink larger circle more
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink the larger circle more to preserve total radius
                    if radii[i] >= radii[j]:
                        # Circle i shrinks by most of the overlap
                        shrink_i = (sum_r - dist) * 0.9999
                        radii[i] = max(radii[i] - shrink_i, 1e-5)
                        # Small shrink on j for safety margin
                        radii[j] = max(radii[j] * 0.99999, 1e-5)
                    else:
                        shrink_j = (sum_r - dist) * 0.9999
                        radii[j] = max(radii[j] - shrink_j, 1e-5)
                        radii[i] = max(radii[i] * 0.99999, 1e-5)
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

def _vectorized_refine(pos, radii, time_limit=3.0):
    """
    Fast vectorized refinement: repeatedly push overlapping circles apart
    and grow radii. Uses numpy for speed.
    """
    start = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    T = 0.02
    cooling = 0.995
    rng = np.random.RandomState(42)
    
    while time.time() - start < time_limit:
        # Vectorized overlap detection and force calculation
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0.0, sum_r - dist)
        
        # Force direction (normalized)
        inv_dist = np.where(dist > 1e-10, 1.0/dist, 0.0)
        fx = dx * inv_dist * overlap * 3.0
        fy = dy * inv_dist * overlap * 3.0
        
        # Sum forces on each circle
        forces_x = np.sum(fx, axis=1)
        forces_y = np.sum(fy, axis=1)
        
        # Boundary forces
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            if x < r: forces_x[i] += (r - x) * 8
            if x > 1-r: forces_x[i] -= (x - (1-r)) * 8
            if y < r: forces_y[i] += (r - y) * 8
            if y > 1-r: forces_y[i] -= (y - (1-r)) * 8
        
        # Apply with noise
        noise_x = rng.normal(0, T, N)
        noise_y = rng.normal(0, T, N)
        pos[:, 0] += forces_x * 0.005 + noise_x * 0.005
        pos[:, 1] += forces_y * 0.005 + noise_y * 0.005
        pos[:, 0] = np.clip(pos[:, 0], 0.01, 0.99)
        pos[:, 1] = np.clip(pos[:, 1], 0.01, 0.99)
        
        # Grow radii
        for i in range(N):
            max_r = min(pos[i, 0], 1-pos[i, 0], pos[i, 1], 1-pos[i, 1])
            for j in range(N):
                if i != j:
                    d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                    max_r = min(max_r, d - radii[j])
            radii[i] = max(max_r * 0.995, 1e-5)
        
        T *= cooling
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Use more of the 150s budget
    
    # Strategy: Multi-start with different initializations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Perturbed random (4 variations with different seeds)
    for i in range(4):
        pos_p, rad_p = _initial_random(100 + i * 137)
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
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: Coarse optimization with low penalty
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(penalty_stages[0],),
                              options={'maxiter': 1200, 'ftol': 1e-10, 'gtol': 1e-6})
            x0 = result.x
        except Exception:
            pass
        
        # Stage 2: Medium penalty
        if time.time() < deadline - 25:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 1200, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: High penalty for final polish
        if time.time() < deadline - 30:
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
        
        # Powell polish for first two inits
        if idx < 2 and time.time() < deadline - 35:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='Powell', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 600, 'ftol': 1e-14})
                if result.fun < _objective(x0, penalty_stages[2]) - 1e-12:
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

    # Vectorized force-directed anneal for additional exploration
    if time.time() < deadline - 25:
        try:
            pos_anneal, radii_anneal = _vectorized_refine(best_pos, best_radii, time_limit=3.0)
            pos_pp, radii_pp = _post_process(pos_anneal, radii_anneal)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final high-precision polish on the best result
    if time.time() < deadline - 12:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 2500, 'ftol': 1e-15, 'gtol': 1e-12})
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
