"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
_DEADLINE_MARGIN = 5.0  # seconds before kill to stop

def _force_directed_anneal(pos, radii, n_steps=2000, t_start=0.05, t_end=0.001, seed=42):
    """
    Force-directed simulated annealing for circle packing.
    Circles repel each other and walls, with a cooling schedule.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    
    for step in range(n_steps):
        # Cooling schedule
        progress = step / max(n_steps - 1, 1)
        temp = t_start * (t_end / t_start) ** progress
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Wall forces
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            # Right wall
            if x > 1.0 - r:
                forces[i, 0] -= (x - (1.0 - r)) * 10.0
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            # Top wall
            if y > 1.0 - r:
                forces[i, 1] -= (y - (1.0 - r)) * 10.0
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq + 1e-12)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Overlapping: strong repulsion
                    overlap = min_dist - dist
                    force_mag = overlap * 5.0
                    fx = (dx / dist) * force_mag
                    fy = (dy / dist) * force_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 2.0:
                    # Near overlap: moderate repulsion to encourage separation
                    force_mag = (min_dist * 2.0 - dist) * 0.5
                    fx = (dx / dist) * force_mag
                    fy = (dy / dist) * force_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Add noise for annealing
        noise = rng.normal(0, temp, (N, 2))
        
        # Update positions
        pos += forces * 0.1 + noise
        
        # Keep within bounds
        for i in range(N):
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r, 1.0 - r)
            pos[i, 1] = np.clip(pos[i, 1], r, 1.0 - r)
        
        # Occasionally try to increase radii of circles that have room
        if step % 50 == 0 and temp > t_end * 5:
            for i in range(N):
                # Check max possible radius at current position
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                for j in range(N):
                    if j != i:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        max_r = min(max_r, dist - radii[j])
                
                if max_r > radii[i] * 1.05:
                    # Try to increase radius
                    new_r = min(max_r * 0.999, radii[i] * 1.02)
                    # Check if this increase causes overlaps
                    valid = True
                    for j in range(N):
                        if j != i:
                            dx = pos[i, 0] - pos[j, 0]
                            dy = pos[i, 1] - pos[j, 1]
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist < new_r + radii[j]:
                                valid = False
                                break
                    if valid:
                        radii[i] = new_r
    
    return pos, radii

def _initial_lattice():
    """
    Generate an initial packing based on a hexagonal lattice pattern.
    This provides a structurally diverse starting point.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing parameters
    r_base = 0.12
    dx_hex = 2 * r_base
    dy_hex = np.sqrt(3) * r_base
    
    k = 0
    row = 0
    col = 0
    while k < N:
        x = 0.5 + col * dx_hex * 0.95
        y = 0.5 + row * dy_hex * 0.95
        if col % 2 == 1:
            y += dy_hex * 0.5
        
        # Check if within bounds
        if x - r_base >= 0 and x + r_base <= 1 and y - r_base >= 0 and y + r_base <= 1:
            pos[k] = [x, y]
            radii[k] = r_base
            k += 1
        
        col += 1
        if col > 5:
            col = 0
            row += 1
            if row > 5:
                # Fallback: place remaining circles randomly
                for rem in range(k, N):
                    pos[rem] = [0.5 + random.uniform(-0.3, 0.3), 0.5 + random.uniform(-0.3, 0.3)]
                    radii[rem] = 0.08
                break
    
    # Post-process to ensure validity
    pos, radii = _post_process(pos, radii)
    return pos, radii

# EVOLVE-BLOCK-START

def _soft_min_penalty(values, scale=1.0):
    """
    Smooth approximation of sum of positive parts using soft-min (log-sum-exp).
    For a vector v, approximates sum(max(0, v_i)) smoothly.
    Uses the identity: max(0, v) ≈ (v + softplus(v) - softplus(-v)) / 2 is not quite right.
    Better: use the fact that for overlap d = r_i + r_j - dist, we want to penalize max(0, d).
    A smooth approximation: d * sigmoid(d/scale) works well for small overlaps.
    Alternatively, use: (d + sqrt(d^2 + eps^2) - eps) / 2 which is smooth and ≈ max(0,d).
    """
    # Smooth max(0, x) using: (x + sqrt(x^2 + eps^2) - eps) / 2
    eps = 1e-6
    return 0.5 * (values + np.sqrt(values * values + eps * eps) - eps)

def _objective(x, penalty_weight=500.0, smooth_scale=0.01):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    Uses smooth penalties based on soft-max approximations.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii_safe = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (Smooth)
    # Violation: circle extends beyond wall
    left_viol = radii_safe - pos[:, 0]      # positive if x < r
    right_viol = pos[:, 0] - (1.0 - radii_safe)  # positive if x > 1-r
    bottom_viol = radii_safe - pos[:, 1]
    top_viol = pos[:, 1] - (1.0 - radii_safe)
    
    # Use smooth max(0, v) penalty
    boundary_penalty = np.sum(_soft_min_penalty(left_viol)) + \
                       np.sum(_soft_min_penalty(right_viol)) + \
                       np.sum(_soft_min_penalty(bottom_viol)) + \
                       np.sum(_soft_min_penalty(top_viol))
    
    # 2. Pairwise Overlap Violations (Vectorized, Smooth)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii_safe[:, np.newaxis] + radii_safe[np.newaxis, :]
    
    # Overlap amount: positive when circles overlap
    # overlap = min_dist - dist, we want to penalize max(0, overlap)
    # To avoid sqrt, work in a way that's still smooth
    # Actually, let's compute dist properly for accuracy
    dist = np.sqrt(dist_sq + 1e-12)
    
    overlap = min_dist - dist  # positive when overlapping
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Smooth penalty for overlaps
    overlap_penalty = np.sum(_soft_min_penalty(overlap[mask]))
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii_safe) + penalty_weight * total_penalty

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

def _post_process(pos, radii, shrink_factor=0.9999):
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
                radii[i] = max(max_r * shrink_factor, 1e-6)
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
                    k = dist / (sum_r + 1e-12) * shrink_factor
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _vectorized_refine(pos, radii, n_iter=50, shrink_factor=0.99999):
    """
    Vectorized iterative refinement: repeatedly project positions and radii
    to satisfy constraints, then try to grow radii where possible.
    This is a local improvement that works well after gradient-based optimization.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(n_iter):
        # Step 1: Enforce boundary constraints
        for i in range(N):
            max_r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r_wall:
                radii[i] = max_r_wall * shrink_factor
        
        # Step 2: Enforce pairwise constraints (iterative)
        for _ in range(10):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    sum_r = radii[i] + radii[j]
                    if dist < sum_r:
                        # Reduce radii to eliminate overlap
                        k = dist / (sum_r + 1e-12) * shrink_factor
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        changed = True
            if not changed:
                break
        
        # Step 3: Try to grow radii where there's room
        for i in range(N):
            # Max radius allowed by walls
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            
            # Max radius allowed by other circles
            for j in range(N):
                if j != i:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, dist - radii[j])
            
            # Grow if possible
            if max_r > radii[i] * 1.001:
                new_r = min(max_r * 0.99999, radii[i] * 1.01)
                # Verify no overlap with new radius
                valid = True
                for j in range(N):
                    if j != i:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < new_r + radii[j] - 1e-9:
                            valid = False
                            break
                if valid:
                    radii[i] = new_r
    
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    This creates a valid, dense configuration that serves as a strong starting point.
    Uses a combination of grid sampling and random sampling for better coverage.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        # Try to find a position for the next circle
        # We will try to maximize the radius of the new circle at a given position
        # by checking distance to all existing circles and walls.
        
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 3000 random points for better coverage
        for _ in range(3000):
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
                # If we place circle i at (x,y) with radius r, and circle j has radius r_j:
                # dist >= r + r_j  =>  r <= dist - r_j
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            # Fallback: place tiny circle at center
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97  # Slight shrink to ensure no overlap in initial state
        
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    # Force validity by shrinking
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - _DEADLINE_MARGIN)
    
    def _remaining():
        return deadline - time.time()
    
    # Strategy: Multi-start with different initializations
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice-based initialization
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Perturbed Best Fit (2 variations for more diversity, save time)
    for i in range(2):
        pos_p, rad_p = _initial_random(100 + i)
        inits.append((pos_p, rad_p))
        
    # 4. One more random with different seed
    pos_r, rad_r = _initial_random(999)
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
        
    # Multi-stage optimization with increasing penalty weight
    penalty_stages = [50.0, 200.0, 1000.0, 5000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if _remaining() < 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: Coarse optimization with low penalty
        if _remaining() > 30:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[0],),
                                  options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 2: Medium penalty
        if _remaining() > 40:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[1],),
                                  options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
                x0 = result.x
            except Exception:
                pass
        
        # Stage 3: High penalty for refinement
        if _remaining() > 50:
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
                x0 = result.x
            except Exception:
                pass
            
            x_opt = x0
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            # Apply vectorized refinement for local improvement
            pos_ref, radii_ref = _vectorized_refine(pos_opt, radii_opt, n_iter=30)
            pos_pp, radii_pp = _post_process(pos_ref, radii_ref)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Force-directed annealing as an alternative optimization path (only for first 2 inits)
        if _remaining() > 35 and idx <= 1:
            try:
                pos_fd, radii_fd = _force_directed_anneal(pos_init, rad_init, 
                                                           n_steps=1000, 
                                                           t_start=0.03, 
                                                           t_end=0.0005,
                                                           seed=1000 + idx)
                pos_fd_ref, radii_fd_ref = _vectorized_refine(pos_fd, radii_fd, n_iter=20)
                pos_fd_pp, radii_fd_pp = _post_process(pos_fd_ref, radii_fd_ref)
                score_fd = np.sum(radii_fd_pp)
                
                if _is_valid(pos_fd_pp, radii_fd_pp) and score_fd > best_score:
                    best_score = score_fd
                    best_pos = pos_fd_pp.copy()
                    best_radii = radii_fd_pp.copy()
            except Exception:
                pass
        
        # If this was the first init, do a Powell polish on the best result
        if idx == 0 and _remaining() > 40:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            try:
                result = minimize(_objective, x0, method='Powell', bounds=bounds,
                                  args=(penalty_stages[2],),
                                  options={'maxiter': 500, 'ftol': 1e-14})
                if result.fun < _objective(x0, penalty_stages[2]) - 1e-12:
                    x_opt = result.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                    pos_ref, radii_ref = _vectorized_refine(pos_opt, radii_opt, n_iter=30)
                    pos_pp, radii_pp = _post_process(pos_ref, radii_ref)
                    score = np.sum(radii_pp)
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass

    # Final high-precision polish on the best result
    if _remaining() > 15:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(penalty_stages[3],),
                              options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            # Vectorized refinement + post-process
            pos_ref, radii_ref = _vectorized_refine(pos_opt, radii_opt, n_iter=50)
            pos_pp, radii_pp = _post_process(pos_ref, radii_ref)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
        
        # Additional force-directed polish on the best result
        if _remaining() > 10:
            try:
                pos_fd2, radii_fd2 = _force_directed_anneal(best_pos, best_radii,
                                                             n_steps=800,
                                                             t_start=0.01,
                                                             t_end=0.0001,
                                                             seed=9999)
                pos_fd2_ref, radii_fd2_ref = _vectorized_refine(pos_fd2, radii_fd2, n_iter=30)
                pos_fd2_pp, radii_fd2_pp = _post_process(pos_fd2_ref, radii_fd2_ref)
                score_fd2 = np.sum(radii_fd2_pp)
                
                if _is_valid(pos_fd2_pp, radii_fd2_pp) and score_fd2 > best_score:
                    best_score = score_fd2
                    best_pos = pos_fd2_pp.copy()
                    best_radii = radii_fd2_pp.copy()
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
