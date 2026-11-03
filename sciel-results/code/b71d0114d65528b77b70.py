"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the common radius r for given centers."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    # Distance from center c to wall is min(0.5 - c_k, c_k + 0.5) for each axis k
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _surrogate_objective_and_grad(centers, beta):
    """
    Compute the negative of the LSE surrogate for the minimum radius and its gradient.
    Used for the final polish stage.
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_diffs = diff[iu]
    
    t_pairs = -0.5 * pair_dists
    
    # 2. Wall terms
    wall_terms = np.zeros((n, 6))
    wall_terms[:, 0] = centers[:, 0] - 0.5
    wall_terms[:, 1] = -centers[:, 0] - 0.5
    wall_terms[:, 2] = centers[:, 1] - 0.5
    wall_terms[:, 3] = -centers[:, 1] - 0.5
    wall_terms[:, 4] = centers[:, 2] - 0.5
    wall_terms[:, 5] = -centers[:, 2] - 0.5
    
    t_walls = wall_terms.flatten()
    all_terms = np.concatenate([t_pairs, t_walls])
    
    max_term = np.max(all_terms)
    exp_terms = np.exp(beta * (all_terms - max_term))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    obj = (1.0 / beta) * (np.log(sum_exp) + beta * max_term)
    
    w_pairs = weights[:len(t_pairs)]
    unit_vecs = pair_diffs / pair_dists[:, np.newaxis]
    
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, 0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    
    w_walls = weights[len(t_pairs):]
    w_walls_reshaped = w_walls.reshape(n, 6)
    
    grad[:, 0] += w_walls_reshaped[:, 0] - w_walls_reshaped[:, 1]
    grad[:, 1] += w_walls_reshaped[:, 2] - w_walls_reshaped[:, 3]
    grad[:, 2] += w_walls_reshaped[:, 4] - w_walls_reshaped[:, 5]
    
    return obj, grad

def _dynamics_simulate(centers, max_time=10.0, seed=0):
    """
    Simulate rigid-body dynamics with repulsive forces and damping.
    Forces:
    - Pairwise: F_ij = k * (c_i - c_j) / ||c_i - c_j||^3  (Repulsive, ~1/r^2 magnitude)
    - Wall:     F_wall = k_wall * (c - boundary) if close to wall (Soft repulsion)
    
    Integrator: Velocity Verlet with damping.
    """
    rng = np.random.RandomState(seed)
    c = centers.copy()
    v = np.zeros_like(c)
    
    # Parameters
    dt = 0.005
    damping = 0.95 # Strong damping to settle quickly
    k_pair = 1.0
    k_wall = 5.0 # Stronger wall force to keep them in
    
    # Precompute indices
    iu = np.triu_indices(N, k=1)
    
    start_time = time.time()
    
    # Initial force calculation
    def calc_forces(pos):
        f = np.zeros_like(pos)
        # Pairwise
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        
        # F_ij = k * diff / dist^3
        # Only upper triangle needed for pairs, but we can compute full matrix and take half?
        # No, force on i from j is opposite to force on j from i.
        # Let's compute vectorized for pairs.
        pair_diffs = diff[iu]
        pair_dists = dists[iu]
        pair_dists_safe = np.where(pair_dists < 1e-6, 1e-6, pair_dists)
        
        # Force magnitude on i from j: k / d^2. Direction: (c_i - c_j)/d.
        # Vector: k * (c_i - c_j) / d^3
        forces = k_pair * pair_diffs / (pair_dists_safe ** 3)[:, np.newaxis]
        
        i_idx, j_idx = iu
        np.add.at(f, i_idx, forces)
        np.add.at(f, j_idx, -forces)
        
        # Wall forces
        # If c_x > 0.5 - margin, push left. If c_x < -0.5 + margin, push right.
        # Use a soft spring: F = -k_wall * (c - bound) if out of bounds? 
        # Better: Always apply a force that pushes away from the nearest wall if within a certain distance.
        # Or simply: F_wall = -k_wall * (pos - center_of_box) ? No, that's central.
        # Let's use: F_x = -k_wall * pos_x if pos_x > 0? No.
        # If pos_x > 0.5, F_x = -k_wall * (pos_x - 0.5).
        # If pos_x < -0.5, F_x = -k_wall * (pos_x + 0.5).
        # This keeps them inside. But we want them to pack, so they should be pushed away from walls only if too close?
        # Actually, the "box" constraint in packing is hard. In dynamics, we can use soft walls.
        # If they touch the wall, they should bounce or stay.
        # Let's use: F_wall = -k_wall * (pos - clamp(pos, -0.5, 0.5)) ? No.
        # Let's use: if pos > 0.5, F = -k*(pos-0.5). if pos < -0.5, F = -k*(pos+0.5).
        # This allows them to go slightly out if forces are strong, but damping will settle them.
        
        # Simpler: Hard clip positions after integration, and add a small restoring force if they hit the wall.
        # Let's just use the restoring force.
        
        # x-axis
        mask_pos_x = pos[:, 0] > 0.5
        f[mask_pos_x, 0] -= k_wall * (pos[mask_pos_x, 0] - 0.5)
        mask_neg_x = pos[:, 0] < -0.5
        f[mask_neg_x, 0] -= k_wall * (pos[mask_neg_x, 0] + 0.5)
        
        mask_pos_y = pos[:, 1] > 0.5
        f[mask_pos_y, 1] -= k_wall * (pos[mask_pos_y, 1] - 0.5)
        mask_neg_y = pos[:, 1] < -0.5
        f[mask_neg_y, 1] -= k_wall * (pos[mask_neg_y, 1] + 0.5)
        
        mask_pos_z = pos[:, 2] > 0.5
        f[mask_pos_z, 2] -= k_wall * (pos[mask_pos_z, 2] - 0.5)
        mask_neg_z = pos[:, 2] < -0.5
        f[mask_neg_z, 2] -= k_wall * (pos[mask_neg_z, 2] + 0.5)
        
        return f
    
    f = calc_forces(c)
    
    while time.time() < start_time + max_time:
        # Velocity Verlet
        # v_half = v + 0.5 * dt * f
        # c_new = c + dt * v_half
        # f_new = calc_forces(c_new)
        # v_new = v_half + 0.5 * dt * f_new
        # v = v_new * damping
        
        v_half = v + 0.5 * dt * f
        c_new = c + dt * v_half
        f_new = calc_forces(c_new)
        v_new = v_half + 0.5 * dt * f_new
        v = v_new * damping
        
        c = c_new
        f = f_new
        
        # Optional: Hard clip to prevent runaway
        # c = np.clip(c, -0.6, 0.6)
        
    return c

def _generate_grid_start():
    """Generate a cubic grid start."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Strategy:
    # 1. Generate several initial configurations (grid, random, fcc-like)
    # 2. For each, run dynamics simulation for a short time to "shake out"
    # 3. Polish the best few with L-BFGS
    
    initializations = []
    
    # Grid
    initializations.append(_generate_grid_start() + np.random.RandomState(1).randn(N, 3) * 0.01)
    
    # Random
    initializations.append(np.random.RandomState(2).uniform(-0.4, 0.4, (N, 3)))
    
    # FCC-like (approx)
    # Just use another random with different seed
    initializations.append(np.random.RandomState(3).uniform(-0.4, 0.4, (N, 3)))
    
    # BCC-like?
    # 3x3x3 BCC = 27*2 = 54. Take 50.
    # Lattice points at (0,0,0) and (0.5,0.5,0.5) in unit cell.
    # Scale to fit in [-0.5, 0.5].
    # 3x3x3 cells.
    # Points:
    bcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Center of cell
                cx = (i + 0.5) / 3.0 - 0.5
                cy = (j + 0.5) / 3.0 - 0.5
                cz = (k + 0.5) / 3.0 - 0.5
                # Corner (0,0,0) relative to cell?
                # Actually, BCC is simpler:
                # Let's just stick to the above 3 inits.
                
    candidates = []
    
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
            
        # Run dynamics
        # Use a moderate time budget for each
        t_sim = min(10.0, (deadline - time.time()) / 3.0)
        
        sim_centers = _dynamics_simulate(init, max_time=t_sim, seed=idx)
        
        # Clip to ensure validity
        sim_centers = np.clip(sim_centers, -0.5, 0.5)
        
        r = _compute_radius(sim_centers)
        if r > best_radius:
            best_radius = r
            best_centers = sim_centers.copy()
            
        candidates.append(sim_centers)
        
    # Polish the best candidates
    # Sort candidates by radius
    if candidates:
        scored_candidates = [( _compute_radius(c), c) for c in candidates]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take top 3
        for r, c in scored_candidates[:3]:
            if time.time() > deadline - 10:
                break
                
            # L-BFGS Polish
            x0 = c.flatten()
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            def fun_and_grad(x):
                cc = x.reshape(N, 3)
                val, grad = _surrogate_objective_and_grad(cc, beta=100.0)
                return val, grad.flatten()
            
            try:
                res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, 
                               options={'maxiter': 200, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
                best_c = res.x.reshape(N, 3)
                best_c = np.clip(best_c, -0.5, 0.5)
                
                r_polished = _compute_radius(best_c)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = best_c.copy()
            except Exception:
                pass
    
    if best_centers is None:
        best_centers = _generate_grid_start()
        
    return np.clip(best_centers, -0.5, 0.5)
# EVOLVE-BLOCK-END
