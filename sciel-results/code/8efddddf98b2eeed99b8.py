"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _force_directed_anneal(pos, radii, start_time, deadline, seed=42):
    """
    Force-directed annealing: treat circles as particles with repulsive forces
    from each other and attractive forces toward optimal positions, with
    simulated annealing to escape local minima.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Parameters
    T0 = 0.1  # Initial temperature
    T_end = 0.001
    cooling_rate = 0.9995
    steps_per_iter = 10
    dt = 0.001
    
    T = T0
    n_steps = 0
    
    while T > T_end and time.time() < deadline - 5:
        for _ in range(steps_per_iter):
            n_steps += 1
            if time.time() > deadline - 5:
                break
            
            forces = np.zeros_like(pos)
            
            # 1. Boundary forces (push away from walls)
            # Left wall
            left_dist = pos[:, 0] - radii
            forces[:, 0] += np.where(left_dist < 0, -1.0 * left_dist, 0.0)
            # Right wall
            right_dist = (1.0 - pos[:, 0]) - radii
            forces[:, 0] += np.where(right_dist < 0, -1.0 * right_dist, 0.0)
            # Bottom wall
            bottom_dist = pos[:, 1] - radii
            forces[:, 1] += np.where(bottom_dist < 0, -1.0 * bottom_dist, 0.0)
            # Top wall
            top_dist = (1.0 - pos[:, 1]) - radii
            forces[:, 1] += np.where(top_dist < 0, -1.0 * top_dist, 0.0)
            
            # 2. Pairwise repulsion (Hooke's law for overlaps)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        # Overlap: repulsive force proportional to overlap
                        overlap = min_dist - dist
                        force_mag = 10.0 * overlap
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # 3. Attractive force toward center of mass (to prevent drift)
            com = pos.mean(axis=0)
            center_force = -0.01 * (pos - com)
            forces += center_force
            
            # 4. Random perturbation (simulated annealing)
            noise = T * rng.randn(*pos.shape)
            forces += noise
            
            # 5. Update positions
            pos += dt * forces
            
            # 6. Clip to valid range
            for i in range(N):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                if max_r < 1e-4:
                    # Move to a random valid position
                    pos[i] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
                    radii[i] = 0.05
                elif radii[i] > max_r:
                    radii[i] = max_r * 0.999
            
            # 7. Try to grow radii
            for i in range(N):
                # Check if circle can grow
                can_grow = True
                max_possible_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                
                # Check against other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r_j = dist - radii[j]
                    max_possible_r = min(max_possible_r, max_r_j)
                
                if max_possible_r > radii[i]:
                    # Grow by a small amount
                    growth = min(0.001, (max_possible_r - radii[i]) * 0.5)
                    radii[i] += growth
            
            # 8. Evaluate and update best
            if _is_valid(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        
        # Cool down
        T *= cooling_rate
    
    return best_pos, best_radii, best_score

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    # Reshape x into positions (N, 2) and radii (N,)
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    # 1. Boundary Violations (Smooth Squared Penalty)
    # Circle i is at (x_i, y_i) with radius r_i.
    # Violation if x_i < r_i (left), x_i > 1-r_i (right), etc.
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Pairwise Overlap Violations (Vectorized)
    # We need distance between every pair (i, j).
    # dist_sq[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    # Required distance: r_i + r_j
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # Overlap exists if dist_sq < min_dist_sq
    # We want a smooth penalty. 
    # If dist_sq >= min_dist_sq, penalty = 0.
    # If dist_sq < min_dist_sq, penalty = (min_dist_sq - dist_sq)^2 / (4 * min_dist_sq)
    # This is 0 at the boundary and has 0 derivative at the boundary.
    
    # Only consider upper triangle to avoid double counting, though summing all is fine if we divide by 2.
    # For efficiency, let's just sum all and divide by 2, or use a mask.
    # Masking is cleaner.
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Calculate overlap amount in squared distance space
    # Note: If min_dist_sq is 0 (radii 0), we have div by zero. 
    # We clamped radii to 1e-6, so min_dist_sq > 0.
    
    overlap_term = np.zeros_like(dist_sq)
    # We only compute where mask is true to save some ops? 
    # Actually, np.where is fast enough.
    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    
    # Avoid division by zero if radii are very small (though clamped)
    denom = np.maximum(min_dist_sq, 1e-12)
    
    smooth_overlap = (violation ** 2) / (4.0 * denom)
    
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
    # Vectorized distance check
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Check if any pair violates dist < min_dist
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist (since both positive)
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
    
    for _ in range(20):
        valid = True
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            
            # Check against walls
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        # Find min clearance
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down both radii proportionally to fit
                    # We want r_i' + r_j' <= dist
                    # Let r_i' = r_i * k, r_j' = r_j * k
                    # k * (r_i + r_j) = dist => k = dist / (r_i + r_j)
                    k = dist / (sum_r + 1e-12) * 0.999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_model():
    """
    Construct a geometrically inspired initial layout.
    Strategy: 
    1. Place 4 circles in corners.
    2. Place remaining 22 in a hexagonal-like grid in the center.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 1. Corners
    # Let's start them slightly away from the absolute corner to avoid singularity issues in gradients if they are exactly at 0,0 with r>0
    corner_r = 0.1
    corners = [
        (0.1, 0.1),
        (0.9, 0.1),
        (0.1, 0.9),
        (0.9, 0.9)
    ]
    for i in range(4):
        pos[i] = corners[i]
        radii[i] = corner_r
        
    # 2. Center Grid
    # We have 22 circles left.
    # A 5x4 grid = 20 circles. Plus 2 extra.
    # Let's make a 5x5 grid for indices 4..24 (21 circles) and 1 extra at 25.
    # Actually, 4 + 22 = 26.
    # Let's use a hexagonal pattern for the center.
    
    # Simpler: 5x5 grid of small circles, then optimizer expands.
    # But to help the optimizer, let's make the center ones smaller so they don't overlap corners initially.
    
    # Let's place 22 circles in a 5x5 grid (minus 1 center? no, just 5x5 = 25 spots, we have 22).
    # Let's use a 5x5 grid for indices 4..24 (21 circles) and put the last one (25) in the center of the remaining gap or just another spot.
    
    # Better: 
    # Indices 4..8: Row y=0.2
    # Indices 9..13: Row y=0.4
    # Indices 14..18: Row y=0.6
    # Indices 19..23: Row y=0.8
    # This is 4 rows of 5 = 20 circles.
    # Plus 2 extra circles (24, 25).
    
    r_center = 0.08
    y_coords = [0.2, 0.4, 0.6, 0.8]
    x_coords = [0.2, 0.4, 0.6, 0.8, 1.0-0.2] # 0.2, 0.4, 0.6, 0.8, 0.8
    
    idx = 4
    for y in y_coords:
        for x in x_coords:
            if idx < N-2: # Leave last 2 for extras
                pos[idx] = [x, y]
                radii[idx] = r_center
                idx += 1
                
    # Now idx should be 4 + 20 = 24.
    # We have indices 24 and 25 left.
    # Place them in the "holes" of the grid or just center.
    # Let's put them at (0.5, 0.3) and (0.5, 0.7) roughly, or just (0.5, 0.5) split.
    # Actually, (0.5, 0.5) is the center of the 5x5 grid.
    # The grid points are 0.2, 0.4, 0.6, 0.8.
    # Midpoints are 0.3, 0.5, 0.7.
    # Let's place 24 at (0.5, 0.3) and 25 at (0.5, 0.7).
    # Or (0.3, 0.5) and (0.7, 0.5).
    # Let's try (0.5, 0.5) is crowded.
    # Let's place 24 at (0.3, 0.5) and 25 at (0.7, 0.5).
    
    pos[24] = [0.3, 0.5]
    radii[24] = 0.07
    pos[25] = [0.7, 0.5]
    radii[25] = 0.07
    
    # Random jitter to break symmetry
    rng = np.random.RandomState(42)
    pos += rng.uniform(-0.01, 0.01, pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    pos, radii = _initial_model()
    
    # Run force-directed annealing first to find a good starting point
    pos_fd, radii_fd, score_fd = _force_directed_anneal(pos, radii, start_time, deadline)
    
    # Use the better of initial or force-directed as starting point for L-BFGS
    if score_fd > np.sum(radii):
        pos, radii = pos_fd, radii_fd
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds:
    # x, y in [0.01, 0.99] to keep them strictly inside for the optimizer's sake, 
    # though the penalty handles it. Tighter bounds help convergence.
    # r in [0.01, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Step 1: L-BFGS-B
    # We want to maximize sum of radii.
    # The objective function includes a penalty. We need to choose the weight carefully.
    # If weight is too high, it minimizes penalty but doesn't grow radii much.
    # If weight is too low, it grows radii but overlaps.
    # 500.0 was used in the good seed. Let's stick with 500.0.
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          args=(500.0,),
                          options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < _objective(x0, 500.0) - 1e-10:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")
        
    # Step 2: Powell for fine tuning (derivative free, robust to non-smoothness in penalty)
    if time.time() < deadline - 10:
        # Re-initialize from the best found
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              args=(500.0,),
                              options={'maxiter': 3000, 'ftol': 1e-12})
            
            if result.fun < _objective(x0, 500.0) - 1e-10:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception as e:
            print(f"Powell failed: {e}")
            
    # Step 3: Final Polish with Nelder-Mead or another L-BFGS pass with tighter tolerance
    if time.time() < deadline - 5:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            # Use a slightly lower penalty weight for final push to expand radii
            # if the penalty is already very small
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              args=(200.0,), # Lower weight to allow more growth if penalty is low
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < _objective(x0, 200.0) - 1e-10:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception as e:
            print(f"Final Polish failed: {e}")
    
    # Step 4: Another round of force-directed annealing from the polished solution
    if time.time() < deadline - 5:
        try:
            pos_fd2, radii_fd2, score_fd2 = _force_directed_anneal(
                best_pos, best_radii, start_time, deadline, seed=123
            )
            if score_fd2 > best_score:
                best_score = score_fd2
                best_pos = pos_fd2.copy()
                best_radii = radii_fd2.copy()
        except Exception as e:
            print(f"Second FD anneal failed: {e}")

    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # If for some reason the result is invalid, fallback to a very safe simple grid
    if not _is_valid(best_pos, best_radii):
         # Fallback: 5x5 grid of r=0.095 + 1 small circle
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
