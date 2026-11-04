"""
Optimized packer for 100 equal spheres in a unit cube.
Uses smooth soft-min objective for global optimization,
target-radius violation minimization for squeezing,
and contact-graph refinement for final polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Minimum distance to boundary
    min_boundary = HALF - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_radius(x, beta):
    """
    Smooth surrogate for the minimum radius using soft-min.
    x is a flat array of length 3N.
    beta is the sharpness parameter (higher = closer to true min).
    Returns (energy, gradient) where energy = -soft_min_radius.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e10)  # Large value for diagonal
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)  # shape (N, 3)
    
    # Collect all "radius candidates"
    # Pairwise: dists[i,j]/2 for i<j
    # Wall: wall_dists[i,k]
    
    # For soft-min, we use: min(x) ≈ -(1/beta) * log(sum(exp(-beta*x)))
    # We want to maximize the min, so minimize -soft_min
    
    # Build array of all radius candidates
    # Pairwise (upper triangle)
    iu = np.triu_indices(N, k=1)
    pair_radii = dists[iu] / 2.0  # shape (M,)
    
    # Wall radii
    wall_radii = wall_dists.flatten()  # shape (3N,)
    
    # All radii
    all_radii = np.concatenate([pair_radii, wall_radii])
    
    # Soft-min: -(1/beta) * log(sum(exp(-beta * r)))
    # To avoid overflow, use log-sum-exp trick
    max_r = np.max(all_radii)
    shifted = -beta * (all_radii - max_r)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_r - (1.0/beta) * log_sum
    
    # Gradient of -soft_min w.r.t. x
    # d(-soft_min)/dx_i = (1/beta) * sum_j [ exp(-beta*(r_j - max_r)) / sum_k exp(-beta*(r_k - max_r)) ] * (-beta * dr_j/dx_i)
    # = - sum_j [ w_j * dr_j/dx_i ]
    # where w_j = exp(-beta*(r_j - max_r)) / sum_k exp(-beta*(r_k - max_r))
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))  # shape (M+3N,)
    
    grad = np.zeros_like(c)
    
    # Pairwise gradient contribution
    # For pair (i,j) with radius d/2, weight w_ij:
    # d/dx_i (d/2) = (x_i - x_j) / (2*d)
    # So contribution to grad_i from pair (i,j): -w_ij * (x_i - x_j) / (2*d)
    # And contribution to grad_j: -w_ij * (x_j - x_i) / (2*d) = +w_ij * (x_i - x_j) / (2*d)
    
    # Vectorized: for each pair (i,j), add -w_ij/d_ij * (x_i - x_j) to grad_i
    # and +w_ij/d_ij * (x_i - x_j) to grad_j
    
    # Build weight matrix for pairs
    pair_weights = weights[:len(pair_radii)]  # shape (M,)
    
    # For each pair (i,j), compute contribution
    # diff[i,j] = x_i - x_j, dists[i,j] = d
    # Contribution to grad_i: -w_ij * diff[i,j] / (2 * d_ij)
    # Contribution to grad_j: +w_ij * diff[i,j] / (2 * d_ij)
    
    # Create weight matrix (only upper triangle)
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    
    # safe_dists for division
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    
    # grad_i = -sum_j w_ij * diff[i,j] / (2 * safe_dists[i,j])
    # = -0.5 * sum_j w_ij * diff[i,j] / safe_dists[i,j]
    
    unit_diffs = diff / safe_dists[:, :, np.newaxis]
    
    # grad_pair[i] = -0.5 * sum_j w_mat[i,j] * unit_diffs[i,j]
    grad_pair = -0.5 * np.sum(w_mat[:, :, np.newaxis] * unit_diffs, axis=1)
    
    # Wall gradient contribution
    # For wall distance r_ik = 0.5 - |c_ik|, weight w_ik:
    # d/dc_ik (0.5 - |c_ik|) = -sign(c_ik)
    # Contribution to grad_ik: -w_ik * (-sign(c_ik)) = w_ik * sign(c_ik)
    
    wall_weights = weights[len(pair_radii):].reshape(N, 3)
    signs = np.sign(c)
    grad_wall = wall_weights * signs
    
    total_grad = (grad_pair + grad_wall).flatten()
    energy = -soft_min_val
    
    return energy, total_grad

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
    Energy is sum of squared violations of the constraints:
    1. Pairwise distance >= 2 * target_r
    2. Wall distance >= target_r
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise Part ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Active pairs: those closer than 2 * target_r
    active = dists < 2.0 * target_r
    # Ensure diagonal is not active
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d) for active pairs, 0 otherwise
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    # Energy is sum of squared violations (0.5 factor for clean gradient)
    energy_pair = 0.5 * np.sum(violations**2)
    
    # Gradient calculation for pairs
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    wall_dists = 0.5 - np.abs(c)
    
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize_soft_min(init_pts, beta_start=10.0, beta_end=100.0, max_iter=500, time_limit=30.0):
    """
    Optimize using soft-min radius objective with annealing.
    """
    t_start = time.time()
    x = init_pts.flatten().copy()
    
    for beta in np.linspace(beta_start, beta_end, 5):
        if time.time() - t_start > time_limit:
            break
        
        def objective(x):
            if time.time() - t_start > time_limit:
                raise TimeoutError()
            return _soft_min_radius(x, beta)
        
        try:
            result = minimize(
                objective,
                x,
                method='BFGS',
                jac=True,
                options={'maxiter': max_iter, 'gtol': 1e-10, 'disp': False}
            )
            x = result.x
        except (TimeoutError, Exception):
            break
    
    opt_pts = x.reshape(N, 3)
    return opt_pts

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-10, time_limit=50.0):
    """
    Run BFGS optimization with a time limit.
    Returns (optimized_pts, success).
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def _contact_graph_refine(points, target_r, max_iter=200, time_limit=10.0):
    """
    Refine points by only moving those involved in close contacts.
    """
    t_start = time.time()
    pts = points.copy()
    
    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break
        
        # Find close pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Active pairs: distance < 2*target_r * 1.05
        active = dists < 2.0 * target_r * 1.05
        active[np.diag_indices_from(active)] = False
        
        # Find indices of active points
        active_indices = np.where(active.any(axis=1))[0]
        
        if len(active_indices) < 2:
            break
        
        # Only optimize active points
        active_pts = pts[active_indices].copy()
        
        def sub_objective(x_sub):
            if time.time() - t_start > time_limit:
                raise TimeoutError()
            # Reconstruct full points
            full_pts = pts.copy()
            full_pts[active_indices] = x_sub.reshape(-1, 3)
            return _energy_and_grad(full_pts.flatten(), target_r)
        
        try:
            result = minimize(
                sub_objective,
                active_pts.flatten(),
                method='BFGS',
                jac=True,
                options={'maxiter': 20, 'gtol': 1e-10, 'disp': False}
            )
            pts[active_indices] = result.x.reshape(-1, 3)
        except (TimeoutError, Exception):
            break
    
    return pts

def _init_fcc_like():
    """
    Initialize points in a structure resembling FCC packing.
    """
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution
    points = _init_fcc_like()
    best_points = points.copy()
    best_r = _compute_radius(best_points)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Soft-min global optimization
    if time.time() < deadline - 80.0:
        # Try multiple initializations with noise
        for trial in range(3):
            if time.time() > deadline - 70.0:
                break
            
            noise_scale = 0.01 * (trial + 1)
            noisy_init = np.clip(points + rng.normal(0, noise_scale, size=(N, 3)), -0.49, 0.49)
            
            remaining_time = deadline - (time.time() - start_time) - 10.0
            if remaining_time < 5.0:
                break
            
            opt_pts = _optimize_soft_min(
                noisy_init, 
                beta_start=5.0, 
                beta_end=50.0, 
                max_iter=200, 
                time_limit=remaining_time
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_points = opt_pts.copy()
    
    # Phase 2: Direct BFGS refinement with target radius
    current_pts = best_points.copy()
    initial_targets = [0.112, 0.114, 0.116, 0.118]
    
    for target_r in initial_targets:
        if time.time() > deadline - 40.0:
            break
        
        noise = rng.normal(0, 0.003, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining_time = deadline - (time.time() - start_time) - 5.0
        if remaining_time < 2.0:
            break
        
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            target_r, 
            max_iter=1500, 
            gtol=1e-10,
            time_limit=remaining_time
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_points = opt_pts.copy()
                current_pts = opt_pts.copy()
    
    # Phase 3: Squeeze phase with smaller increments
    num_squeezes = 20
    for i in range(num_squeezes):
        if time.time() > deadline - 20.0:
            break
        
        success_improved = False
        increments = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for inc in increments:
            if time.time() > deadline - 15.0:
                break
            
            target_r = best_r * (1.0 + inc)
            
            noise = rng.normal(0, 0.001, size=(N, 3))
            init_pts = np.clip(best_points + noise, -0.49, 0.49)
            
            remaining_time = deadline - (time.time() - start_time) - 3.0
            if remaining_time < 1.0:
                break
            
            opt_pts, success = _optimize_bfgs(
                init_pts, 
                target_r, 
                max_iter=1000, 
                gtol=1e-10,
                time_limit=remaining_time
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_points = opt_pts.copy()
                    success_improved = True
                    break
        
        if not success_improved and rng.random() < 0.4:
            noise = rng.normal(0, 0.005, size=(N, 3))
            init_pts = np.clip(best_points + noise, -0.49, 0.49)
            target_r = best_r * 1.002
            remaining_time = deadline - (time.time() - start_time) - 2.0
            if remaining_time > 1.0:
                opt_pts, success = _optimize_bfgs(
                    init_pts, 
                    target_r, 
                    max_iter=800, 
                    gtol=1e-10,
                    time_limit=remaining_time
                )
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _compute_radius(opt_pts)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_points = opt_pts.copy()
    
    # Phase 4: Contact graph refinement
    if time.time() < deadline - 10.0:
        remaining_time = deadline - (time.time() - start_time) - 5.0
        if remaining_time > 5.0:
            refined_pts = _contact_graph_refine(
                best_points, best_r, 
                max_iter=50, time_limit=remaining_time
            )
            if np.all(np.isfinite(refined_pts)):
                refined_pts = np.clip(refined_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_refined = _compute_radius(refined_pts)
                if r_refined > best_r:
                    best_r = r_refined
                    best_points = refined_pts.copy()
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        target_r = best_r * 1.0001
        remaining_time = deadline - (time.time() - start_time) - 2.0
        if remaining_time > 3.0:
            polished_pts, success = _optimize_bfgs(
                best_points, target_r, 
                max_iter=3000, gtol=1e-12,
                time_limit=remaining_time
            )
            if np.all(np.isfinite(polished_pts)):
                polished_pts = np.clip(polished_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_polished = _compute_radius(polished_pts)
                if r_polished > best_r:
                    best_r = r_polished
                    best_points = polished_pts.copy()
    
    return best_points

# EVOLVE-BLOCK-END
