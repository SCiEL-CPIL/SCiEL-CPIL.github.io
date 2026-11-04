"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time


def _compute_score(centres):
    """Compute the packing score: min of half min pairwise distance and min wall distance."""
    if centres.shape[0] != N:
        return -np.inf
    if not np.all(np.isfinite(centres)):
        return -np.inf
    # Wall constraint
    wall_dists = np.min(np.abs(centres), axis=1)
    wall_radius = CUBE_HALF - np.max(wall_dists)
    # Pairwise constraint
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    pair_radius = min_pair_dist / 2.0
    return min(wall_radius, pair_radius)


def _project_to_box(centres):
    """Project points onto the cube [-0.5, 0.5]^3."""
    return np.clip(centres, -CUBE_HALF, CUBE_HALF)


def _alternate_projection(centres, r_target, max_iter=5000, tol=1e-10):
    """
    Alternating projection to find a configuration where all spheres of radius r_target
    fit inside the cube and don't overlap.
    """
    centres = centres.copy()
    for it in range(max_iter):
        # Project onto box constraint
        centres = _project_to_box(centres)
        
        # Project onto pairwise separation constraint
        # For each pair that is too close, push them apart
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        # Find pairs that violate the constraint
        violation_mask = dists < 2 * r_target
        if not np.any(violation_mask):
            break
        
        # For each violating pair, compute the correction
        # We do a simple gradient-like step
        # For efficiency, we only update pairs that are violating
        violating_indices = np.where(violation_mask)
        
        # Accumulate forces
        forces = np.zeros_like(centres)
        for i, j in zip(violating_indices[0], violating_indices[1]):
            diff = centres[i] - centres[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-15:
                # Random perturbation
                diff = np.random.randn(3) * 1e-8
                dist = np.linalg.norm(diff)
            # Push apart
            # The amount to push is (2*r_target - dist) / 2 for each
            push_amount = (2 * r_target - dist) / (2 * dist)
            forces[i] += push_amount * diff
            forces[j] -= push_amount * diff
        
        centres += forces
        
        # Re-project to box
        centres = _project_to_box(centres)
    
    return centres


def _local_search(centres, max_iter=2000, lr_initial=0.01):
    """
    Local search to maximize the minimum radius.
    Uses a smooth surrogate: minimize -log(sum(exp(-d_ij/t)) + exp(-wall_dist/t))
    or simply use gradient ascent on the min radius with a soft-min.
    """
    centres = centres.copy()
    t = 0.01  # temperature for soft-min
    lr = lr_initial
    
    for it in range(max_iter):
        # Compute distances
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        pair_radius = min_pair_dist / 2.0
        
        # Wall distances
        wall_dists = CUBE_HALF - np.max(np.abs(centres), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        current_radius = min(pair_radius, min_wall_dist)
        
        # Soft-min for gradient computation
        # We want to maximize the min of (pair_radius, min_wall_dist)
        # Use log-sum-exp trick for soft-min
        # soft_min(a, b) = -log(exp(-a/t) + exp(-b/t)) * t
        # But we need gradients w.r.t. centres
        
        # Let's compute gradients numerically or analytically
        # Analytical approach:
        # The gradient of min_{i<j} |c_i - c_j| / 2 w.r.t. c_k
        # is in the direction of the closest pair involving k
        
        # Find the closest pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i_min, j_min = min_pair_idx
        
        # Find the closest wall point
        wall_idx = np.argmin(wall_dists)
        
        # Gradient of pair distance w.r.t. centres[i_min] and centres[j_min]
        diff = centres[i_min] - centres[j_min]
        dist_val = np.linalg.norm(diff)
        if dist_val > 1e-15:
            grad_pair_i = diff / (2 * dist_val)
            grad_pair_j = -diff / (2 * dist_val)
        else:
            grad_pair_i = np.zeros(3)
            grad_pair_j = np.zeros(3)
        
        # Gradient of wall distance w.r.t. centres[wall_idx]
        # wall_dist = 0.5 - max(|c_x|, |c_y|, |c_z|)
        # The gradient is in the direction that increases the min wall distance
        # i.e., towards the center along the axis with max |c|
        wall_coords = centres[wall_idx]
        max_abs_idx = np.argmax(np.abs(wall_coords))
        grad_wall = np.zeros(3)
        grad_wall[max_abs_idx] = -np.sign(wall_coords[max_abs_idx])
        
        # Combine gradients
        # We want to move in the direction that increases the minimum radius
        # If pair_radius < min_wall_dist, focus on pair
        # If min_wall_dist < pair_radius, focus on wall
        # Otherwise, combine
        
        grad = np.zeros_like(centres)
        if pair_radius <= min_wall_dist:
            # Focus on increasing pair distance
            grad[i_min] += grad_pair_i
            grad[j_min] += grad_pair_j
            # Also add small wall gradient to keep inside
            for k in range(N):
                wall_c = centres[k]
                max_abs_k = np.argmax(np.abs(wall_c))
                grad[k] += 0.1 * (-np.sign(wall_c[max_abs_k]) * np.eye(3)[max_abs_k])
        else:
            # Focus on increasing wall distance
            grad[wall_idx] += grad_wall
            # Also add small pair gradient
            grad[i_min] += 0.1 * grad_pair_i
            grad[j_min] += 0.1 * grad_pair_j
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Update
        centres += lr * grad
        
        # Project to box
        centres = _project_to_box(centres)
        
        # Decay learning rate
        lr *= 0.999
        
        # Check convergence
        if it % 100 == 0:
            new_radius = _compute_score(centres)
            if new_radius > current_radius + 1e-8:
                pass  # Good, continue
            elif lr < 1e-6:
                break
    
    return centres


def _random_init(seed=None):
    """Generate a random initial configuration."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, size=(N, 3))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_centres = None
    best_score = -np.inf
    
    # Start with the simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centres = grid[:N].copy()
    
    # Try local search from the grid
    result = _local_search(initial_centres, max_iter=5000, lr_initial=0.005)
    score = _compute_score(result)
    if score > best_score:
        best_score = score
        best_centres = result.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_centres
    
    # Try multiple random restarts
    for restart in range(50):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        random_centres = _random_init(seed=42 + restart)
        
        # First, try to fix overlaps with alternating projection
        # Start with a small radius and increase
        # Estimate initial radius
        dists = cdist(random_centres, random_centres)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        wall_dists = CUBE_HALF - np.max(np.abs(random_centres), axis=1)
        min_wall = np.min(wall_dists)
        initial_r = min(min_dist / 2, min_wall)
        
        # Use alternating projection to find a feasible configuration for a slightly larger radius
        target_r = initial_r * 1.05
        if target_r > 0.01:
            fixed_centres = _alternate_projection(random_centres, target_r, max_iter=2000)
            score_fixed = _compute_score(fixed_centres)
            
            if score_fixed > best_score:
                best_score = score_fixed
                best_centres = fixed_centres.copy()
        
        # Then do local search
        result = _local_search(random_centres, max_iter=3000, lr_initial=0.003)
        score = _compute_score(result)
        if score > best_score:
            best_score = score
            best_centres = result.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
    
    # Final polish: try to increase radius slightly
    if best_centres is not None and time.time() - start_time < deadline - 5:
        # Try alternating projection with a slightly larger radius
        current_r = _compute_score(best_centres)
        for trial in range(10):
            if time.time() - start_time > deadline - 5:
                break
            target_r = current_r * (1.01 + trial * 0.005)
            polished = _alternate_projection(best_centres, target_r, max_iter=1000)
            score_polished = _compute_score(polished)
            if score_polished > current_r:
                current_r = score_polished
                best_centres = polished.copy()
            else:
                break
        
        # Final local search
        result = _local_search(best_centres, max_iter=2000, lr_initial=0.001)
        score = _compute_score(result)
        if score > _compute_score(best_centres):
            best_centres = result
    
    # Ensure validity
    if best_centres is None:
        # Fallback to grid
        best_centres = initial_centres
    
    # Final check and projection
    best_centres = _project_to_box(best_centres)
    
    return best_centres
# EVOLVE-BLOCK-END
