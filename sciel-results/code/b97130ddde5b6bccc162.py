"""
Optimized packer for 100 equal spheres in a unit cube.
Uses jellyfish parameterization with global anchors and local lattice,
combined with BFGS optimization and incremental radius squeezing.
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
    # d/dx_i [ 0.5 * (2r - ||x_i - x_j||)^2 ]
    # = (2r - d) * d/dx_i (2r - d)
    # = (2r - d) * (-(x_i - x_j) / d)
    # = - (2r - d) / d * (x_i - x_j)
    # Note: violations[i,j] = 2r - d. So we need - violations[i,j] / d * (x_i - x_j)
    # To avoid division by zero, we use safe_dists
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights[i,j] = violations[i,j]
    # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    # Distance to each wall
    # Walls are at -0.5 and 0.5 for each dimension
    # dist_wall_k = 0.5 - |c_k|
    wall_dists = 0.5 - np.abs(c)
    
    # Violations: target_r - wall_dist if positive
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    # Gradient for walls
    # d/dc [ (r - (0.5 - |c|))^2 ]
    # = 2 * (r - (0.5 - |c|)) * d/dc (0.5 - |c|)
    # = 2 * wall_violation * (-sign(c))
    signs = np.sign(c)
    # Handle c=0 case: sign(0)=0, which is fine, gradient is 0 at center
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _jellyfish_forward(params, anchors):
    """
    Convert jellyfish parameters to actual sphere centers.
    params: (N, 3) array of local offsets from base lattice
    anchors: (8, 3) array of corner anchor positions
    Returns: (N, 3) array of sphere centers
    """
    # Base 5x5x5 lattice (but we only use 100 points, so 4x5x5)
    nx, ny, nz = 4, 5, 5
    base_xs = np.linspace(-0.3, 0.3, nx)
    base_ys = np.linspace(-0.3, 0.3, ny)
    base_zs = np.linspace(-0.3, 0.3, nz)
    
    # Create base lattice points
    base_pts = np.array([(x, y, z) for z in base_zs for y in base_ys for x in base_xs])
    
    # For each point, compute weighted blend with nearest anchors
    centers = np.zeros((N, 3))
    for i in range(N):
        bp = base_pts[i]
        # Find 2 nearest anchors (corners)
        dists_to_anchors = np.linalg.norm(anchors - bp, axis=1)
        nearest_2 = np.argsort(dists_to_anchors)[:2]
        
        # Weights based on inverse distance
        d1, d2 = dists_to_anchors[nearest_2[0]], dists_to_anchors[nearest_2[1]]
        w1 = 1.0 / (d1 + 1e-6)
        w2 = 1.0 / (d2 + 1e-6)
        w_sum = w1 + w2
        w1, w2 = w1 / w_sum, w2 / w_sum
        
        # Blend: 70% local offset from base, 30% anchor influence
        local_pt = bp + params[i]
        anchor_pt = w1 * anchors[nearest_2[0]] + w2 * anchors[nearest_2[1]]
        centers[i] = 0.7 * local_pt + 0.3 * anchor_pt
    
    return centers

def _jellyfish_grad(params, anchors, target_r):
    """
    Compute gradient of energy w.r.t. jellyfish parameters.
    Uses automatic differentiation via finite differences for simplicity.
    """
    eps = 1e-7
    grad = np.zeros_like(params)
    
    base_centers = _jellyfish_forward(params, anchors)
    base_energy, _ = _energy_and_grad(base_centers.flatten(), target_r)
    
    for i in range(N):
        for j in range(3):
            params[i, j] += eps
            perturbed = _jellyfish_forward(params, anchors)
            pert_energy, _ = _energy_and_grad(perturbed.flatten(), target_r)
            params[i, j] -= eps
            grad[i, j] = (pert_energy - base_energy) / eps
    
    return grad

def _optimize_jellyfish(params_init, anchors_init, target_r, max_iter=500, time_limit=30.0):
    """
    Optimize jellyfish parameters using BFGS.
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        params = x[:3*N].reshape(N, 3)
        anchors = x[3*N:].reshape(8, 3)
        centers = _jellyfish_forward(params, anchors)
        return _energy_and_grad(centers.flatten(), target_r)
    
    def grad(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        params = x[:3*N].reshape(N, 3)
        anchors = x[3*N:].reshape(8, 3)
        # Use finite difference for anchor gradient, analytical for params
        grad_params = _jellyfish_grad(params, anchors, target_r)
        
        # Finite difference for anchors
        grad_anchors = np.zeros_like(anchors)
        eps = 1e-7
        base_energy, _ = objective(x)
        for a in range(8):
            for d in range(3):
                x[a*3 + d] += eps
                e_plus = objective(x)[0]
                x[a*3 + d] -= eps
                grad_anchors[a, d] = (e_plus - base_energy) / eps
        
        return np.concatenate([grad_params.flatten(), grad_anchors.flatten()])
    
    x_init = np.concatenate([params_init.flatten(), anchors_init.flatten()])
    
    try:
        result = minimize(
            objective,
            x_init,
            method='BFGS',
            jac=grad,
            options={'maxiter': max_iter, 'gtol': 1e-8, 'disp': False}
        )
        opt_params = result.x[:3*N].reshape(N, 3)
        opt_anchors = result.x[3*N:].reshape(8, 3)
        opt_centers = _jellyfish_forward(opt_params, opt_anchors)
        return opt_centers, opt_params, opt_anchors, result.success
    except TimeoutError:
        return _jellyfish_forward(params_init, anchors_init), params_init, anchors_init, False
    except Exception:
        return _jellyfish_forward(params_init, anchors_init), params_init, anchors_init, False

def _init_jellyfish():
    """
    Initialize jellyfish parameters and anchors.
    """
    # Start with regular grid-like anchors
    anchors = np.array([
        [-0.4, -0.4, -0.4], [0.4, -0.4, -0.4],
        [-0.4, 0.4, -0.4], [0.4, 0.4, -0.4],
        [-0.4, -0.4, 0.4], [0.4, -0.4, 0.4],
        [-0.4, 0.4, 0.4], [0.4, 0.4, 0.4]
    ])
    
    # Initial params: small random offsets
    rng = np.random.RandomState(42)
    params = rng.normal(0, 0.01, size=(N, 3))
    
    return params, anchors

def _polish_high_precision(points, target_r, max_iter=2000, time_limit=20.0):
    """
    Final high-precision polish using tighter tolerance.
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-12, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return points, False
    except Exception:
        return points, False

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
    
    # Phase 1: Jellyfish global optimization
    if time.time() < deadline - 60.0:
        params_init, anchors_init = _init_jellyfish()
        
        # Try a few target radii with jellyfish
        jellyfish_targets = [0.105, 0.110, 0.112, 0.114]
        
        for target_r in jellyfish_targets:
            if time.time() > deadline - 50.0:
                break
            
            remaining_time = deadline - (time.time() - start_time) - 10.0
            if remaining_time < 5.0:
                break
            
            # Add noise to break symmetry
            params_noisy = params_init + rng.normal(0, 0.005, size=(N, 3))
            anchors_noisy = anchors_init + rng.normal(0, 0.01, size=(8, 3))
            anchors_noisy = np.clip(anchors_noisy, -0.45, 0.45)
            
            opt_centers, opt_params, opt_anchors, success = _optimize_jellyfish(
                params_noisy, anchors_noisy, target_r, 
                max_iter=300, time_limit=remaining_time
            )
            
            if np.all(np.isfinite(opt_centers)):
                opt_centers = np.clip(opt_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _compute_radius(opt_centers)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_points = opt_centers.copy()
                    # Update anchors for next iteration
                    anchors_init = opt_anchors
    
    # Phase 2: Direct BFGS refinement
    current_pts = best_points.copy()
    initial_targets = [0.112, 0.114, 0.116, 0.118]
    
    for target_r in initial_targets:
        if time.time() > deadline - 30.0:
            break
        
        # Add small noise to break symmetry
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
            # Clip to ensure validity
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_points = opt_pts.copy()
                current_pts = opt_pts.copy()
    
    # Phase 3: Squeeze phase with smaller increments
    num_squeezes = 15
    for i in range(num_squeezes):
        if time.time() > deadline - 15.0:
            break
        
        success_improved = False
        increments = [0.0015, 0.0008, 0.0004, 0.0002]
        
        for inc in increments:
            if time.time() > deadline - 10.0:
                break
            
            target_r = best_r * (1.0 + inc)
            
            noise = rng.normal(0, 0.0015, size=(N, 3))
            init_pts = np.clip(best_points + noise, -0.49, 0.49)
            
            remaining_time = deadline - (time.time() - start_time) - 3.0
            if remaining_time < 1.0:
                break
            
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
                    success_improved = True
                    break
        
        if not success_improved and rng.random() < 0.3:
            # Random restart
            noise = rng.normal(0, 0.008, size=(N, 3))
            init_pts = np.clip(best_points + noise, -0.49, 0.49)
            target_r = best_r * 1.003
            remaining_time = deadline - (time.time() - start_time) - 2.0
            if remaining_time > 1.0:
                opt_pts, success = _optimize_bfgs(
                    init_pts, 
                    target_r, 
                    max_iter=600, 
                    gtol=1e-10,
                    time_limit=remaining_time
                )
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _compute_radius(opt_pts)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_points = opt_pts.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        target_r = best_r * 1.0001
        remaining_time = deadline - (time.time() - start_time) - 2.0
        if remaining_time > 3.0:
            polished_pts, success = _polish_high_precision(
                best_points, target_r, 
                max_iter=3000, time_limit=remaining_time
            )
            if np.all(np.isfinite(polished_pts)):
                polished_pts = np.clip(polished_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_polished = _compute_radius(polished_pts)
                if r_polished > best_r:
                    best_r = r_polished
                    best_points = polished_pts.copy()
    
    return best_points

# EVOLVE-BLOCK-END
