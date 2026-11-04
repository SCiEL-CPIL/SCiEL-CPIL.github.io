"""
Optimized packing of 50 spheres in a unit cube.
Strategy:
1. Initialize with diverse configurations (FCC, BCC, HCP, layered, random).
2. Run damped molecular dynamics with adaptive shaking.
3. Polish with aggressive beta annealing using soft-min objective.
4. Active-set refinement on critical constraints.
5. Final high-precision polish pass.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
EPS = 1e-12

def _compute_score(centers):
    """Compute the packing radius for a set of centers."""
    centers = np.clip(centers, -CUBE_HALF + EPS, CUBE_HALF - EPS)
    
    # Wall distance
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_obj_and_grad(params, beta, centers_shape=(N, 3)):
    """
    Objective and gradient: minimize -soft_min(distances)
    Returns (value, gradient)
    """
    centers = params.reshape(centers_shape)
    centers = np.clip(centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    # Precomputed indices for upper triangle
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d_full = np.sqrt(np.sum(diff**2, axis=-1))
    d_pairs = d_full[tri_i, tri_j] / 2.0
    
    # Wall distances
    d_x_pos = CUBE_HALF - centers[:, 0]
    d_x_neg = centers[:, 0] + CUBE_HALF
    d_y_pos = CUBE_HALF - centers[:, 1]
    d_y_neg = centers[:, 1] + CUBE_HALF
    d_z_pos = CUBE_HALF - centers[:, 2]
    d_z_neg = centers[:, 2] + CUBE_HALF
    
    wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
    all_dists = np.concatenate([d_pairs, wall_dists])
    
    # Soft minimum calculation using logsumexp for numerical stability
    neg_beta_g = -beta * all_dists
    log_sum_exp = logsumexp(neg_beta_g)
    soft_min_val = -log_sum_exp / beta
    
    # Weights for gradient
    weights = np.exp(neg_beta_g - log_sum_exp)
    
    # Gradient computation
    grad_centers = np.zeros_like(centers)
    
    # Pairwise gradient
    diff_ij = centers[tri_i] - centers[tri_j]
    d_ij = d_full[tri_i, tri_j]
    safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
    unit_vecs = diff_ij / safe_d[:, np.newaxis]
    w_pair = weights[:n_pairs] * 0.5 / safe_d
    contrib = w_pair[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_centers, tri_i, contrib)
    np.add.at(grad_centers, tri_j, -contrib)
    
    # Wall gradients
    ws = n_pairs
    grad_centers[:, 0] -= weights[ws:ws+N]
    grad_centers[:, 0] += weights[ws+N:ws+2*N]
    grad_centers[:, 1] -= weights[ws+2*N:ws+3*N]
    grad_centers[:, 1] += weights[ws+3*N:ws+4*N]
    grad_centers[:, 2] -= weights[ws+4*N:ws+5*N]
    grad_centers[:, 2] += weights[ws+5*N:ws+6*N]
    
    return -soft_min_val, -grad_centers.flatten()

def _dynamics_phase(centers, time_budget):
    """
    Run a damped molecular dynamics simulation with adaptive shaking.
    """
    positions = centers.copy().astype(np.float64)
    velocities = np.zeros_like(positions)
    
    k_force = 1.0
    damping = 0.95
    dt = 0.005
    shake_interval = 300
    shake_magnitude = 0.05
    step_count = 0
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(positions)
        
        # Pairwise forces with stronger repulsion for close pairs
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Use inverse cube law for repulsion
        inv_r3 = 1.0 / (dists**3)
        np.fill_diagonal(inv_r3, 0.0)
        
        force_mat = k_force * diff * inv_r3[:, :, np.newaxis]
        forces += np.sum(force_mat, axis=1)
        
        # Wall forces
        d_left = np.clip(positions[:, 0] + CUBE_HALF, 1e-6, None)
        d_right = np.clip(CUBE_HALF - positions[:, 0], 1e-6, None)
        d_bottom = np.clip(positions[:, 1] + CUBE_HALF, 1e-6, None)
        d_top = np.clip(CUBE_HALF - positions[:, 1], 1e-6, None)
        d_back = np.clip(positions[:, 2] + CUBE_HALF, 1e-6, None)
        d_front = np.clip(CUBE_HALF - positions[:, 2], 1e-6, None)
        
        forces[:, 0] += k_force / (d_left**2) - k_force / (d_right**2)
        forces[:, 1] += k_force / (d_bottom**2) - k_force / (d_top**2)
        forces[:, 2] += k_force / (d_back**2) - k_force / (d_front**2)
        
        velocities = velocities * damping + forces * dt
        positions = positions + velocities * dt
        positions = np.clip(positions, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        # Adaptive shaking to escape local minima
        step_count += 1
        if step_count % shake_interval == 0:
            # Gradually reduce shake magnitude
            shake_mag = shake_magnitude * max(0.1, 1.0 - step_count / (shake_interval * 10))
            positions += shake_mag * np.random.randn(*positions.shape)
            positions = np.clip(positions, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            velocities *= 0.5  # Dampen velocities after shake
    
    return positions

def _build_fcc_grid():
    """Build FCC-inspired lattice initialization."""
    points = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                points.append([ix*0.25 - 0.5, iy*0.25 - 0.5, iz*0.25 - 0.5])
                if ix < 3: points.append([(ix+0.5)*0.25 - 0.5, iy*0.25 - 0.5, iz*0.25 - 0.5])
                if iy < 3: points.append([ix*0.25 - 0.5, (iy+0.5)*0.25 - 0.5, iz*0.25 - 0.5])
                if iz < 3: points.append([ix*0.25 - 0.5, iy*0.25 - 0.5, (iz+0.5)*0.25 - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5 + 1e-9, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        grid = points[:N]
    else:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        grid = np.vstack([points, extra])
    
    perturbation = 0.02 * np.random.randn(len(grid), 3)
    grid = grid + perturbation
    grid = np.clip(grid, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return grid

def _build_bcc_grid():
    """Build BCC-inspired lattice initialization."""
    points = []
    # Use 3x3x3 BCC = 27 corner + 27 body = 54 points, take 50
    a = 1.0 / 3.0  # lattice parameter for unit cube
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                # Corner points
                x = -0.5 + ix * a
                y = -0.5 + iy * a
                z = -0.5 + iz * a
                points.append([x, y, z])
                # Body center point
                points.append([x + a/2, y + a/2, z + a/2])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5 + 1e-9, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        grid = points[:N]
    else:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        grid = np.vstack([points, extra])
    
    perturbation = 0.02 * np.random.randn(len(grid), 3)
    grid = grid + perturbation
    grid = np.clip(grid, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return grid

def _build_layered_grid():
    """Build layered initialization with alternating offsets."""
    points = []
    n_layers = 5
    points_per_layer = N // n_layers
    remaining = N - points_per_layer * n_layers
    
    for layer in range(n_layers):
        z = -0.4 + layer * 0.2
        n_in_layer = points_per_layer + (1 if layer < remaining else 0)
        
        for i in range(n_in_layer):
            angle = 2 * np.pi * i / n_in_layer + (layer % 2) * np.pi / n_in_layer
            r = 0.35
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y, z])
    
    points = np.array(points[:N])
    perturbation = 0.02 * np.random.randn(len(points), 3)
    points = points + perturbation
    points = np.clip(points, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    return points

def _build_random_grid():
    """Build random initialization."""
    points = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return points

def _active_set_polish(c_init, time_budget):
    """
    Identify binding constraints and polish with a focused smooth optimization.
    Multiple iterations with adaptive tolerance.
    """
    c = c_init.copy()
    r = _compute_score(c)
    tri_i, tri_j = np.triu_indices(N, k=1)
    
    for _ in range(8):
        if time.time() > time_budget:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        r_current = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Adaptive tolerance: tighter as we converge
        tol = 0.002 * r_current + 1e-9
        
        pair_dists = d[tri_i, tri_j] / 2.0
        binding_pairs = np.where(pair_dists <= r_current + tol)[0]
        
        dist_pos = 0.5 - c
        dist_neg = c + 0.5
        
        pos_flat = dist_pos.flatten()
        neg_flat = dist_neg.flatten()
        
        pos_bind_flat = np.where(pos_flat <= r_current + tol)[0]
        neg_bind_flat = np.where(neg_flat <= r_current + tol)[0]
        
        pos_bind_i = pos_bind_flat // 3
        pos_bind_k = pos_bind_flat % 3
        neg_bind_i = neg_bind_flat // 3
        neg_bind_k = neg_bind_flat % 3
        
        n_bind_pos = len(pos_bind_i)
        n_bind_neg = len(neg_bind_i)
        n_bind_pairs = len(binding_pairs)
        
        if n_bind_pairs == 0 and n_bind_pos == 0 and n_bind_neg == 0:
            return c, r_current
        
        def binding_objective_and_grad(c_flat, beta):
            c = c_flat.reshape(N, 3)
            
            if np.any(np.abs(c) > 0.5 + 1e-9):
                return 1e6, np.zeros_like(c_flat)
            
            if n_bind_pairs > 0:
                diff_bind = c[tri_i[binding_pairs]] - c[tri_j[binding_pairs]]
                d_bind = np.linalg.norm(diff_bind, axis=1) / 2.0
            else:
                d_bind = np.array([])
                diff_bind = np.zeros((0, 3))
            
            if n_bind_pos > 0:
                wall_pos = 0.5 - c[pos_bind_i, pos_bind_k]
            else:
                wall_pos = np.array([])
            
            if n_bind_neg > 0:
                wall_neg = c[neg_bind_i, neg_bind_k] + 0.5
            else:
                wall_neg = np.array([])
            
            wall_dists = np.concatenate([wall_pos, wall_neg])
            g_all = np.concatenate([d_bind, wall_dists])
            
            if len(g_all) == 0:
                return 0.0, np.zeros_like(c_flat)
            
            neg_beta_g = -beta * g_all
            log_sum_exp = logsumexp(neg_beta_g)
            val = -log_sum_exp / beta
            
            weights = np.exp(neg_beta_g - log_sum_exp)
            
            grad_c = np.zeros_like(c)
            
            if n_bind_pairs > 0:
                safe_d = np.where(d_bind * 2 > 1e-14, d_bind * 2, 1.0)
                unit_vecs = diff_bind / safe_d[:, np.newaxis]
                w_pair = weights[:n_bind_pairs] * 0.5 / safe_d
                contrib = w_pair[:, np.newaxis] * unit_vecs
                np.add.at(grad_c, tri_i[binding_pairs], contrib)
                np.add.at(grad_c, tri_j[binding_pairs], -contrib)
            
            if n_bind_pos > 0:
                w_pos = weights[n_bind_pairs:n_bind_pairs + n_bind_pos]
                np.add.at(grad_c, (pos_bind_i, pos_bind_k), w_pos[:, np.newaxis])
            
            if n_bind_neg > 0:
                w_neg = weights[n_bind_pairs + n_bind_pos:n_bind_pairs + n_bind_pos + n_bind_neg]
                np.add.at(grad_c, (neg_bind_i, neg_bind_k), -w_neg[:, np.newaxis])
            
            return -val, -grad_c.flatten()
        
        # Try a few different betas for the active set
        for beta in [500, 2000, 10000]:
            if time.time() > time_budget:
                break
            
            try:
                x0 = c.flatten()
                bounds = [(-0.499999, 0.499999)] * (N * 3)
                
                def fun(x):
                    val, grad = binding_objective_and_grad(x, beta)
                    return val
                
                def jac(x):
                    val, grad = binding_objective_and_grad(x, beta)
                    return grad
                
                result = minimize(
                    fun, x0, jac=jac,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 150, 'maxcor': 80}
                )
                
                new_c = result.x.reshape(N, 3)
                new_c = np.clip(new_c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                new_score = _compute_score(new_c)
                
                if new_score > r:
                    c = new_c
                    r = new_score
            except Exception:
                pass
        
        # Update binding constraints for next iteration
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        r_current = min(min_pair_dist / 2.0, min_wall_dist)
        tol = 0.002 * r_current + 1e-9
        
        pair_dists = d[tri_i, tri_j] / 2.0
        binding_pairs = np.where(pair_dists <= r_current + tol)[0]
        
        dist_pos = 0.5 - c
        dist_neg = c + 0.5
        pos_flat = dist_pos.flatten()
        neg_flat = dist_neg.flatten()
        pos_bind_flat = np.where(pos_flat <= r_current + tol)[0]
        neg_bind_flat = np.where(neg_flat <= r_current + tol)[0]
        pos_bind_i = pos_bind_flat // 3
        pos_bind_k = pos_bind_flat % 3
        neg_bind_i = neg_bind_flat // 3
        neg_bind_k = neg_bind_flat % 3
        n_bind_pos = len(pos_bind_i)
        n_bind_neg = len(neg_bind_i)
        n_bind_pairs = len(binding_pairs)
    
    return c, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # 1. Generate multiple diverse initial configurations
    initializers = [
        _build_fcc_grid,
        _build_bcc_grid,
        _build_layered_grid,
        _build_random_grid
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for init_func in initializers:
        if time.time() > deadline - 60.0:
            break
        
        centers = init_func()
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = _build_fcc_grid()
        best_score = _compute_score(best_centers)
    
    # 2. Dynamics phase on best initial configuration
    if time.time() < deadline - 40.0:
        dyn_budget = min(30.0, deadline - time.time() - 40.0)
        dyn_centers = _dynamics_phase(best_centers, dyn_budget)
        dyn_score = _compute_score(dyn_centers)
        
        if dyn_score > best_score:
            best_score = dyn_score
            best_centers = dyn_centers.copy()
    
    # 3. Polish phase with aggressive beta annealing
    if time.time() < deadline - 20.0:
        # Start with moderate beta, increase aggressively
        beta_schedule = [50, 200, 1000, 5000, 20000, 100000, 400000]
        
        for beta in beta_schedule:
            if time.time() > deadline - 10.0:
                break
            
            try:
                start_params = best_centers.flatten()
                
                def fun(x):
                    val, _ = _soft_min_obj_and_grad(x, beta)
                    return val
                
                def jac(x):
                    _, grad = _soft_min_obj_and_grad(x, beta)
                    return grad
                
                result = minimize(
                    fun,
                    start_params,
                    jac=jac,
                    method='L-BFGS-B',
                    bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100, 'maxcor': 50}
                )
                
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
            except Exception:
                pass
    
    # 4. Active-set refinement
    if time.time() < deadline - 10.0:
        try:
            active_budget = deadline - 5.0
            active_centers, active_score = _active_set_polish(best_centers, active_budget)
            
            if active_score > best_score:
                best_score = active_score
                best_centers = active_centers.copy()
        except Exception:
            pass
    
    # 5. Final high-precision polish with very high beta
    if time.time() < deadline - 3.0:
        try:
            start_params = best_centers.flatten()
            final_beta = 1000000
            
            def fun_final(x):
                val, _ = _soft_min_obj_and_grad(x, final_beta)
                return val
            
            def jac_final(x):
                _, grad = _soft_min_obj_and_grad(x, final_beta)
                return grad
            
            result = minimize(
                fun_final,
                start_params,
                jac=jac_final,
                method='L-BFGS-B',
                bounds=[(-CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200, 'maxcor': 100}
            )
            
            final_centers = result.x.reshape(N, 3)
            final_centers = np.clip(final_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            
            final_score = _compute_score(final_centers)
            if final_score > best_score:
                best_score = final_score
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # Final validity check
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_centers

# EVOLVE-BLOCK-START
# The main construct() function is defined above
# EVOLVE-BLOCK-END
