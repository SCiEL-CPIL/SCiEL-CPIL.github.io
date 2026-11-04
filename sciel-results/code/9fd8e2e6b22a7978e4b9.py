import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def _soft_min_obj_grad(x, beta=50.0):
    """
    Soft-min objective using log-sum-exp approximation.
    Maximizes the soft-minimum of all pairwise distances/2 and wall distances.
    """
    pos = x.reshape(-1, 3)
    n = len(pos)
    
    # Compute all pairwise distances / 2
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    # Get upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0  # These are the radii from pairs
    
    # Wall distances
    abs_pos = np.abs(pos)
    max_abs = np.max(abs_pos, axis=1)
    wall_dists = HALF - max_abs
    
    # Combine all "radius" values into one vector
    # We want to maximize the soft-minimum of all these values
    all_vals = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta * v))) / beta
    # This approximates min(v) as beta -> infinity
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_beta_vals - max_neg)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient with respect to x
    # d(soft_min)/dx = (1/beta) * sum_j exp(-beta*v_j) * d(v_j)/dx / sum_j exp(-beta*v_j)
    # But we want to maximize soft_min, so minimize -soft_min
    # d(-soft_min)/dx = -(1/beta) * sum_j exp(-beta*v_j) * d(v_j)/dx / sum_j exp(-beta*v_j)
    
    exp_vals = np.exp(neg_beta_vals - max_neg)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(pos)
    
    # Pairwise gradient contributions
    # v_ij = |pos_i - pos_j| / 2
    # d(v_ij)/d(pos_i) = (pos_i - pos_j) / (2 * |pos_i - pos_j|)
    # d(v_ij)/d(pos_j) = -(pos_i - pos_j) / (2 * |pos_i - pos_j|)
    
    pair_weights = weights[:len(pair_dists)]
    pair_dists_full = dists[i_idx, j_idx]
    
    for k in range(len(pair_dists)):
        if pair_dists_full[k] > 1e-12:
            vec = (pos[i_idx[k]] - pos[j_idx[k]]) / pair_dists_full[k]
            w = pair_weights[k]
            # d(-soft_min)/dx_i += -w * (1/beta) * vec
            # d(-soft_min)/dx_j += -w * (1/beta) * (-vec)
            grad[i_idx[k]] += -w * vec / beta
            grad[j_idx[k]] += w * vec / beta
    
    # Wall gradient contributions
    # v_wall_i = 0.5 - max(|pos_ix|, |pos_iy|, |pos_iz|)
    # d(v_wall_i)/d(pos_ik) = -sign(pos_ik) if k is the argmax component
    
    wall_weights = weights[len(pair_dists):]
    signs = np.sign(pos)
    signs[signs == 0] = 1.0
    
    # For each sphere, find which component is the max
    is_max_comp = (abs_pos == max_abs[:, np.newaxis])
    
    for i in range(n):
        w = wall_weights[i]
        if w > 1e-15:
            for k in range(3):
                if is_max_comp[i, k]:
                    # d(-soft_min)/d(pos_ik) += -w * (1/beta) * (-sign(pos_ik))
                    # = w * sign(pos_ik) / beta
                    grad[i, k] += w * signs[i, k] / beta
    
    return -soft_min_val, grad.flatten()

def _hard_energy_and_grad(x, target_R):
    """
    Hard constraint energy: sum of squared violations of distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = 2.0 * target_R - d_ij
    
    active_mask = v_ij > 0
    if np.any(active_mask):
        v_active = v_ij[active_mask]
        d_active = d_ij[active_mask]
        i_active = i_idx[active_mask]
        j_active = j_idx[active_mask]
        
        energy += np.sum(v_active**2)
        
        factors = -2.0 * v_active / d_active
        vecs = diff[i_active, j_active]
        
        forces_i = factors[:, np.newaxis] * vecs
        forces_j = -forces_i
        
        np.add.at(grad, i_active, forces_i)
        np.add.at(grad, j_active, forces_j)
    
    # Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < target_R
    if np.any(mask_wall):
        v_wall = target_R - wall_dists
        v_wall[~mask_wall] = 0.0
        
        energy += np.sum(v_wall**2)
        
        signs = np.sign(centers)
        signs[signs == 0] = 1.0
        
        is_max = np.isclose(abs_centers, max_abs[:, np.newaxis], rtol=1e-12, atol=1e-12)
        
        grad_wall = 2.0 * v_wall[:, np.newaxis] * signs * is_max
        grad += grad_wall * mask_wall[:, np.newaxis]
        
    return energy, grad.flatten()

def _generate_fcc_init():
    """Generate FCC lattice initialization for 100 spheres."""
    # FCC unit cell has 4 atoms. We need 25 unit cells.
    # Arrange in a 5x5x1 or 5x2x3 grid of unit cells.
    
    # FCC basis positions in unit cell [0,1]^3:
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Try different grid arrangements
    best_init = None
    best_r = 0.0
    
    grids = [(5, 5, 1), (5, 3, 2), (4, 5, 2), (5, 4, 2)]
    
    for nx, ny, nz in grids:
        if nx * ny * nz < 25:
            continue
        
        # Generate FCC points
        points = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    for b in basis:
                        p = np.array([ix + b[0], iy + b[1], iz + b[2]])
                        points.append(p)
        
        points = np.array(points)
        
        # Scale and center to fit in [-0.5, 0.5]^3
        # The FCC lattice spans [0, nx] x [0, ny] x [0, nz]
        # We want to fit this into [0, 1] x [0, 1] x [0, 1]
        scale = 1.0 / max(nx, ny, nz)
        points = points * scale
        
        # Center
        points = points - 0.5
        
        # Keep only first 100 points
        if len(points) >= 100:
            points = points[:100]
        
        # Clip to bounds
        points = np.clip(points, -0.49, 0.49)
        
        r = compute_min_dist(points)
        if r > best_r:
            best_r = r
            best_init = points
    
    return best_init, best_r

def _optimize_soft_min(init_pos, max_iter=200, time_limit=15.0, beta_start=20.0, beta_end=100.0):
    """
    Optimize using soft-min objective with adaptive beta.
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use multiple beta values, increasing from low to high
    betas = [beta_start, 30.0, 50.0, 70.0, beta_end]
    
    current_pos = init_pos.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        def obj_and_grad(x, b=beta):
            return _soft_min_obj_grad(x, beta=b)
        
        try:
            res = minimize(
                obj_and_grad,
                current_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter // len(betas), 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_pos = res.x.reshape(-1, 3)
            if np.all(np.isfinite(new_pos)):
                current_pos = new_pos
        except Exception:
            pass
    
    return current_pos

def _optimize_hard(init_pos, target_R, max_iter=200, time_limit=10.0):
    """
    Optimize using hard constraint energy for a target radius.
    """
    def obj_and_grad(x):
        return _hard_energy_and_grad(x, target_R)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    start_time = time.time()
    try:
        res = minimize(
            obj_and_grad,
            init_pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        return res.x.reshape(-1, 3)
    except Exception:
        return init_pos

def _polish_hard_min(pos, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    High-precision polish pass using the hard-min objective.
    """
    def hard_min_obj_and_grad(flat_pos):
        pos = flat_pos.reshape(N, 3)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        d_walls = HALF - np.abs(pos)
        min_wall_val = np.min(d_walls)
        min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
        wi, wk = min_wall_idx
        
        r_pair = min_pair_val / 2.0
        r_wall = min_wall_val
        
        grad = np.zeros_like(pos)
        
        if r_pair <= r_wall:
            d = min_pair_val
            if d > 1e-12:
                vec = (pos[i] - pos[j]) / d
                grad[i] += 0.5 * vec
                grad[j] += -0.5 * vec
            return -r_pair, -grad.flatten()
        else:
            s = np.sign(pos[wi, wk])
            if s == 0:
                s = 1.0
            grad[wi, wk] = -s
            return -r_wall, -grad.flatten()

    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            hard_min_obj_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        new_pos = res.x.reshape(N, 3)
        if np.all(np.isfinite(new_pos)):
            return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate initial positions
    rng = np.random.default_rng(42)
    
    # Try FCC initialization
    fcc_init, fcc_r = _generate_fcc_init()
    
    # Also try grid initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)
    
    grid_init = make_grid(5, 4, 5)
    grid_r = compute_min_dist(grid_init)
    
    # Choose best initial
    if fcc_r >= grid_r:
        best_pos = fcc_init.copy()
        best_r = fcc_r
    else:
        best_pos = grid_init.copy()
        best_r = grid_r
    
    # Phase 1: Soft-min optimization to get a good starting point
    if time.time() < deadline - 30.0:
        soft_pos = _optimize_soft_min(best_pos, max_iter=300, time_limit=25.0)
        soft_r = compute_min_dist(soft_pos)
        if soft_r > best_r:
            best_r = soft_r
            best_pos = soft_pos
    
    # Phase 2: Incremental radius increase with hard constraints
    current_target = best_r
    attempt_count = 0
    last_improvement_time = start_time
    stagnation_count = 0
    
    while time.time() < deadline - 20.0:
        attempt_count += 1
        time_left = deadline - time.time() - 20.0
        
        if time_left < 5.0:
            break
        
        # Determine target radius
        if time.time() - last_improvement_time > 15.0:
            # Stagnation: try random restart
            stagnation_count += 1
            target_R = best_r * 1.02
            init_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        else:
            # Normal progression
            # Adaptive step size: smaller when close to limit
            if stagnation_count > 3:
                step = 0.0005
            elif stagnation_count > 1:
                step = 0.001
            else:
                step = 0.002
            target_R = best_r + step
            init_pos = best_pos + rng.normal(0, 0.005, size=(N, 3))
            init_pos = np.clip(init_pos, -0.49, 0.49)
        
        # Run optimization
        opt_time = min(12.0, time_left * 0.7)
        
        if opt_time < 2.0:
            break
        
        new_pos = _optimize_hard(init_pos, target_R, max_iter=150, time_limit=opt_time)
        
        achieved_r = compute_min_dist(new_pos)
        
        if achieved_r > best_r:
            best_r = achieved_r
            best_pos = new_pos
            last_improvement_time = time.time()
            stagnation_count = 0
        else:
            # If we didn't improve, try a soft-min pass from this position
            if time.time() < deadline - 25.0 and achieved_r > best_r * 0.95:
                soft_pos = _optimize_soft_min(new_pos, max_iter=100, time_limit=8.0)
                soft_r = compute_min_dist(soft_pos)
                if soft_r > best_r:
                    best_r = soft_r
                    best_pos = soft_pos
                    last_improvement_time = time.time()
                    stagnation_count = 0
    
    # Final polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish_hard_min(best_pos, max_iter=400)
        final_r = compute_min_dist(polished_pos)
        if final_r > best_r:
            best_r = final_r
            best_pos = polished_pos
    
    # Ensure final position is valid
    if not np.all(np.isfinite(best_pos)):
        best_pos = make_grid(5, 4, 5)
    
    return best_pos

# EVOLVE-BLOCK-END
