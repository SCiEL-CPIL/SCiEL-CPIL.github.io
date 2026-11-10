"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance using cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _hexagonal_init():
    """Create a hexagonal packing by selecting N closest points to center from a hex lattice."""
    # Generate a large hexagonal lattice
    i_max = 25
    j_max = 25
    
    points = []
    for j in range(-j_max, j_max + 1):
        for i in range(-i_max, i_max + 1):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2.0
            points.append((x, y))
    
    points = np.array(points)
    
    # Select N points closest to center
    dists_to_center = np.sum(points**2, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    selected_points = points[idx]
    
    # Scale to fit in [-0.5, 0.5] with margin for radius
    max_extent = max(np.max(np.abs(selected_points[:, 0])), 
                     np.max(np.abs(selected_points[:, 1])))
    
    # s * (max_extent + 0.5) <= 0.5  =>  s <= 0.5 / (max_extent + 0.5)
    s = 0.5 / (max_extent + 0.5)
    
    centers = selected_points * s
    
    return centers


def _force_relaxation(centres, target_r, deadline, max_iters=500, learning_rate=0.01):
    """
    Relax the configuration to satisfy constraints for target_r.
    Uses a force-based approach where violations of distance 2*target_r 
    and wall distance target_r generate repulsive forces.
    Returns (success, new_centres).
    """
    centres = centres.copy()
    # Ensure we start valid-ish, clip to bounds
    centres = np.clip(centres, -0.5 + 1e-9, 0.5 - 1e-9)
    
    for it in range(max_iters):
        if time.time() > deadline:
            return False, centres
            
        # Compute forces
        forces = np.zeros_like(centres)
        
        # 1. Pairwise forces
        # Use KDTree for efficiency if N is large, but N=200 is small enough for O(N^2) with NumPy vectorization
        # However, O(N^2) with Python loops is slow. Let's use cdist.
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        # We only care about pairs where dist < 2 * target_r
        # To avoid division by zero and huge forces, we use a soft potential
        # F_ij = k * (2*r - d_ij) * (dir_ij / d_ij) if d_ij < 2*r else 0
        
        # Vectorized calculation
        # Directions
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        # Normalize directions (avoid div by zero)
        norms = dists
        # Safe division
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        dir_vecs = diff / safe_norms[:, :, np.newaxis]
        
        # Strength: proportional to violation
        # violation = max(0, 2*r - d)
        violations = np.maximum(0, 2 * target_r - dists)
        
        # Force magnitude for pair i,j is violations[i,j]
        # Force on i from j is violations[i,j] * dir_vecs[i,j]
        # Sum over j for each i
        # Note: dir_vecs[i,j] is vector from j to i? 
        # diff = c_i - c_j. So dir is from j to i.
        # Force on i should be in direction away from j, which is c_i - c_j. Correct.
        
        pair_forces = np.sum(violations[:, :, np.newaxis] * dir_vecs, axis=1)
        
        # 2. Wall forces
        # Dist to walls
        w_left = 0.5 - centres[:, 0]
        w_right = 0.5 + centres[:, 0]
        w_bottom = 0.5 - centres[:, 1]
        w_top = 0.5 + centres[:, 1]
        
        wall_dists = np.stack([w_left, w_right, w_bottom, w_top], axis=1)
        min_wall_dist = np.min(wall_dists, axis=1)
        
        # Violations: max(0, target_r - w)
        wall_violations = np.maximum(0, target_r - wall_dists)
        
        # Force direction: away from wall
        # If w_left is min, push right (+x)
        # We can compute force component-wise
        # F_x from left wall: if w_left < r, push +x with strength (r - w_left)
        # F_x from right wall: if w_right < r, push -x with strength (r - w_right)
        
        f_x = np.zeros(N)
        f_y = np.zeros(N)
        
        # Left wall (x < -0.5 + r)
        mask_l = w_left < target_r
        f_x[mask_l] += (target_r - w_left[mask_l])
        
        # Right wall (x > 0.5 - r)
        mask_r = w_right < target_r
        f_x[mask_r] -= (target_r - w_right[mask_r])
        
        # Bottom wall (y < -0.5 + r)
        mask_b = w_bottom < target_r
        f_y[mask_b] += (target_r - w_bottom[mask_b])
        
        # Top wall (y > 0.5 - r)
        mask_t = w_top < target_r
        f_y[mask_t] -= (target_r - w_top[mask_t])
        
        wall_forces = np.stack([f_x, f_y], axis=1)
        
        total_forces = pair_forces + wall_forces
        
        # Update positions
        centres += learning_rate * total_forces
        
        # Clip to bounds to prevent escaping
        centres = np.clip(centres, -0.5, 0.5)
        
        # Check convergence: if max force is small, or max violation is small
        max_violation = np.max(violations)
        max_wall_violation = np.max(wall_violations)
        
        if max_violation < 1e-6 and max_wall_violation < 1e-6:
            return True, centres
            
    # Check if it's valid enough
    final_r = _compute_radius(centres)
    if final_r >= target_r * 0.999:
        return True, centres
        
    return False, centres


def _polish(centres, deadline):
    """
    High-precision polish using the 'push most constrained' strategy.
    This is similar to the original local_optimization but refined.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    if best_r <= 0:
        return best_centres, best_r
        
    step_size = best_r * 0.1
    no_improve_count = 0
    stall_limit = 50
    
    for it in range(1000):
        if time.time() > deadline:
            break
            
        # Compute constraints
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)
        
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        pair_constraints = min_pair_dists / 2.0
        constraints = np.minimum(pair_constraints, wall_dists)
        
        idx = np.argmin(constraints)
        
        # Determine direction to push
        if pair_constraints[idx] <= wall_dists[idx]:
            # Find nearest neighbor
            nearest_idx = np.argmin(dists[idx])
            direction = best_centres[idx] - best_centres[nearest_idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm
            
            # Try moving both
            for scale in [1.0, 0.5, 0.25]:
                delta = step_size * scale * direction
                new_centres = best_centres.copy()
                new_centres[idx] += delta
                new_centres[nearest_idx] -= delta
                
                if np.any(new_centres < -0.5 + 1e-10) or np.any(new_centres > 0.5 - 1e-10):
                    continue
                    
                new_r = _compute_radius(new_centres)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_centres
                    no_improve_count = 0
                    break
        else:
            # Push from wall
            # Determine which wall
            dx = np.array([
                0.5 - best_centres[idx, 0],
                0.5 + best_centres[idx, 0],
                0.5 - best_centres[idx, 1],
                0.5 + best_centres[idx, 1]
            ])
            wall_idx = np.argmin(dx)
            direction = np.zeros(2)
            if wall_idx == 0: direction[0] = -1.0
            elif wall_idx == 1: direction[0] = 1.0
            elif wall_idx == 2: direction[1] = -1.0
            else: direction[1] = 1.0
            
            for scale in [1.0, 0.5, 0.25]:
                delta = step_size * scale * direction
                new_centres = best_centres.copy()
                new_centres[idx] += delta
                
                if np.any(new_centres < -0.5 + 1e-10) or np.any(new_centres > 0.5 - 1e-10):
                    continue
                
                new_r = _compute_radius(new_centres)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_centres
                    no_improve_count = 0
                    break
        
        if no_improve_count == 0:
            step_size *= 0.95
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-8 or no_improve_count > stall_limit:
                # Jitter
                if no_improve_count > stall_limit * 2:
                    break
                jitter = np.random.uniform(-step_size * 0.5, step_size * 0.5, size=best_centres.shape)
                new_centres = best_centres + jitter
                new_centres = np.clip(new_centres, -0.499, 0.499)
                new_r = _compute_radius(new_centres)
                if new_r >= best_r - 1e-10:
                    best_centres = new_centres
                    best_r = new_r
                no_improve_count = 0
                step_size = max(0.001, best_r * 0.1)
                
    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # Leave 5s safety margin
    
    best_centres = None
    best_r = -1
    
    # Try multiple random seeds for diversity
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Start with hexagonal packing
        centres = _hexagonal_init()
        
        # Add small random perturbation
        if seed != 42:
            perturbation = np.random.uniform(-0.003, 0.003, size=centres.shape)
            centres += perturbation
            centres = np.clip(centres, -0.499, 0.499)
        
        current_r = _compute_radius(centres)
        
        # Phase 1: Force Relaxation to improve global structure
        # We try to increase the radius incrementally
        # Start with a slightly higher target than current
        target_r = current_r * 1.05
        
        # Run relaxation
        # We don't know the max possible, so we iterate
        for _ in range(5): # Max 5 increments
            if time.time() > deadline:
                break
            success, centres = _force_relaxation(centres, target_r, deadline, max_iters=100, learning_rate=0.005)
            if success:
                current_r = _compute_radius(centres)
                if current_r < target_r:
                    # It didn't quite reach target, but might be better than before
                    if current_r > best_r:
                        best_r = current_r
                        best_centres = centres.copy()
                    break
                else:
                    # It reached target, try higher
                    target_r *= 1.02
                    if current_r > best_r:
                        best_r = current_r
                        best_centres = centres.copy()
            else:
                # Failed to reach target, use the best configuration found during relaxation?
                # _force_relaxation returns the final state.
                current_r = _compute_radius(centres)
                if current_r > best_r:
                    best_r = current_r
                    best_centres = centres.copy()
                break
        
        # Phase 2: Polish
        if time.time() < deadline:
            centres, current_r = _polish(centres, deadline)
            if current_r > best_r:
                best_r = current_r
                best_centres = centres.copy()
        
        # Early stop if we beat the known best
        if best_r > 0.0366:
            break
    
    # Final check
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > 0:
            return best_centres
    
    # Fallback to a valid grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
