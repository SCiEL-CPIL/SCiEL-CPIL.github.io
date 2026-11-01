import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)

def _penalty_objective(centers, target_r, penalty_weight):
    """
    Compute a smooth penalty-based objective that is zero when all constraints
    are satisfied for target_r, and positive otherwise.
    """
    # Wall violations: centers must be at least target_r from each wall
    # i.e., |c_ik| <= 0.5 - target_r
    wall_margin = 0.5 - target_r
    wall_violations = np.maximum(0.0, np.abs(centers) - wall_margin)
    wall_penalty = np.sum(wall_violations ** 2)
    
    # Pair violations: |c_i - c_j| >= 2*target_r for all i<j
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    # For each pair, violation is max(0, 2*target_r - dist)
    pair_violations = np.maximum(0.0, 2.0 * target_r - dists)
    # Only count upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_penalty = np.sum(pair_violations[iu] ** 2)
    
    total_penalty = wall_penalty + pair_penalty
    return penalty_weight * total_penalty

def _penalty_gradient(centers, target_r, penalty_weight):
    """
    Compute gradient of the penalty objective w.r.t. centers.
    """
    grad = np.zeros_like(centers)
    
    # Wall gradient
    wall_margin = 0.5 - target_r
    abs_c = np.abs(centers)
    wall_viol = np.maximum(0.0, abs_c - wall_margin)
    # d/dx of max(0, |x| - m)^2 = 2*max(0,|x|-m)*sign(x)
    grad += 2.0 * penalty_weight * wall_viol * np.sign(centers)
    
    # Pair gradient
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * target_r - dists)
    # For pair (i,j): violation = max(0, 2r - d_ij)
    # d/dc_i of violation^2 = 2*violation * d(violation)/d(c_i)
    # d(violation)/d(c_i) = -d(d_ij)/d(c_i) = -(c_i - c_j)/d_ij when violated
    # So contribution to grad[i] = 2*penalty_weight*violation * (c_i - c_j)/d_ij
    # And to grad[j] = 2*penalty_weight*violation * (c_j - c_i)/d_ij = negative of above
    
    iu = np.triu_indices(N, k=1)
    for i, j in zip(iu[0], iu[1]):
        v = pair_viol[i, j]
        if v > 0:
            d = dists[i, j]
            if d > 1e-12:
                diff = centers[i] - centers[j]
                factor = 2.0 * penalty_weight * v / d
                grad[i] += factor * diff
                grad[j] -= factor * diff
    
    return grad

def _optimize_for_target(init_centers, target_r, deadline, rng):
    """
    Try to find a configuration where all spheres of radius target_r fit.
    Uses gradient descent on the penalty objective with adaptive penalty weight.
    Returns (centers, success) where success means penalty is below threshold.
    """
    c = init_centers.copy()
    c = np.clip(c, -(0.5 - target_r + 0.05), 0.5 - target_r + 0.05)
    
    penalty_weight = 10.0
    lr = 0.005
    success_threshold = 1e-10
    no_improve = 0
    max_no_improve = 500
    
    best_penalty = _penalty_objective(c, target_r, penalty_weight)
    best_c = c.copy()
    
    while time.time() < deadline:
        pen = _penalty_objective(c, target_r, penalty_weight)
        if pen < success_threshold:
            return c, True
        
        grad = _penalty_gradient(c, target_r, penalty_weight)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-10:
            no_improve += 1
        else:
            # Adaptive learning rate
            c_new = c - lr * grad
            c_new = np.clip(c_new, -0.5, 0.5)
            
            pen_new = _penalty_objective(c_new, target_r, penalty_weight)
            
            if pen_new < pen:
                c = c_new
                if pen_new < best_penalty:
                    best_penalty = pen_new
                    best_c = c.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Reduce learning rate or add noise
                lr *= 0.9
                c_new = c + rng.normal(0, 0.005, c.shape)
                c_new = np.clip(c_new, -0.5, 0.5)
                pen_new = _penalty_objective(c_new, target_r, penalty_weight)
                if pen_new < best_penalty:
                    best_penalty = pen_new
                    best_c = c_new.copy()
                c = c_new
                no_improve += 1
        
        # Increase penalty weight periodically to tighten constraints
        if no_improve > max_no_improve:
            penalty_weight *= 2.0
            lr = 0.005
            # Add random perturbation to escape local minima
            c = c + rng.normal(0, 0.02, c.shape)
            c = np.clip(c, -0.5, 0.5)
            best_penalty = _penalty_objective(c, target_r, penalty_weight)
            best_c = c.copy()
            no_improve = 0
        
        # Safety: if penalty weight gets too large, reset
        if penalty_weight > 1e6:
            penalty_weight = 10.0
            c = best_c + rng.normal(0, 0.03, best_c.shape)
            c = np.clip(c, -0.5, 0.5)
            best_penalty = _penalty_objective(c, target_r, penalty_weight)
            best_c = c.copy()
            no_improve = 0
    
    final_penalty = _penalty_objective(best_c, target_r, penalty_weight)
    return best_c, final_penalty < success_threshold

def _init_grid():
    """Grid initialization."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.column_stack((X.ravel(), Y.ravel()))[:N]

def _init_fcc(rng):
    """FCC-like initialization."""
    centers = []
    a = 0.9 / 4.0  # lattice constant
    for i in range(5):
        for j in range(5):
            for k in range(4):
                x = -0.45 + i * a + rng.normal(0, 0.01)
                y = -0.45 + j * a + rng.normal(0, 0.01)
                z = -0.45 + k * a + rng.normal(0, 0.01)
                centers.append([x, y, z])
    # Add FCC offset positions
    for i in range(4):
        for j in range(4):
            for k in range(3):
                x = -0.45 + (i + 0.5) * a + rng.normal(0, 0.01)
                y = -0.45 + (j + 0.5) * a + rng.normal(0, 0.01)
                z = -0.45 + (k + 0.5) * a + rng.normal(0, 0.01)
                centers.append([x, y, z])
    centers = np.array(centers[:N])
    return centers

def _init_random(rng):
    """Random initialization."""
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial radius estimate
    init1 = _init_grid()
    init2 = _init_fcc(rng)
    init3 = _init_random(rng)
    
    r_init1 = _compute_radius(init1)
    r_init2 = _compute_radius(init2)
    r_init3 = _compute_radius(init3)
    
    best_radius = max(r_init1, r_init2, r_init3)
    best_centers = [init1, init2, init3][np.argmax([r_init1, r_init2, r_init3])]
    
    # Binary search on target radius
    # We know the radius is between some lower bound and an upper bound
    lower_bound = best_radius
    upper_bound = 0.15  # Generous upper bound
    
    # First, try to improve from current best using penalty optimization
    # with a target slightly above current best
    
    # Time allocation:
    # 40% for initial optimization attempts
    # 50% for binary search refinement
    # 10% for final polish
    
    t_phase1_end = start_time + (deadline - start_time) * 0.40
    t_phase2_end = start_time + (deadline - start_time) * 0.90
    
    # Phase 1: Try to improve radius from each initialization
    inits = [init1, init2, init3]
    for i, init in enumerate(inits):
        if time.time() > t_phase1_end:
            break
        
        r_cur = _compute_radius(init)
        if r_cur <= best_radius:
            continue
        
        # Try to optimize for target = r_cur * 1.05
        target = r_cur * 1.05
        slice_deadline = min(time.time() + (t_phase1_end - start_time) * 0.3, t_phase1_end - 0.5)
        
        c_opt, success = _optimize_for_target(init, target, slice_deadline, rng)
        if success:
            r_opt = _compute_radius(c_opt)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = c_opt.copy()
        else:
            # Even if not fully successful, the optimized config might be better
            r_opt = _compute_radius(c_opt)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = c_opt.copy()
    
    # Phase 2: Binary search for maximum feasible radius
    # Use the best_centers as starting point for each trial
    
    lo = best_radius
    hi = min(upper_bound, best_radius * 1.3)  # Don't search too far above
    
    # Number of binary search steps based on time available
    max_bs_steps = 8
    
    for step in range(max_bs_steps):
        if time.time() > t_phase2_end - 5.0:
            break
        
        mid = (lo + hi) / 2.0
        # Time per trial
        trial_deadline = time.time() + (t_phase2_end - time.time()) / (max_bs_steps - step) * 0.8
        
        # Try multiple starting points for this target
        found_feasible = False
        for init in [best_centers, _init_grid(), _init_fcc(rng)]:
            if time.time() > trial_deadline:
                break
            
            c_opt, success = _optimize_for_target(init, mid, trial_deadline, rng)
            if success:
                found_feasible = True
                r_opt = _compute_radius(c_opt)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = c_opt.copy()
                break
            else:
                r_opt = _compute_radius(c_opt)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = c_opt.copy()
        
        if found_feasible:
            lo = mid
        else:
            hi = mid
    
    # Phase 3: Final polish - try to squeeze out more radius
    if time.time() < deadline - 2.0:
        target = best_radius * 1.02
        c_opt, success = _optimize_for_target(best_centers, target, deadline - 0.5, rng)
        r_opt = _compute_radius(c_opt)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
    
    return best_centers

# EVOLVE-BLOCK-END
