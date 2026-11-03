import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    if pos.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    # Vectorized pairwise distance calculation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    # Distance to walls: min_k (0.5 - |pos_ik|)
    # This is equivalent to 0.5 - max_k |pos_ik|
    d_walls = HALF - np.max(np.abs(pos), axis=1)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def _softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def energy_and_gradient_soft(flat_pos, target_R, beta):
    """
    Calculates the smooth energy and gradient using softplus penalties.
    Energy = sum(softplus(beta * (2R - d_ij))^2) + sum(softplus(beta * (R - d_wall))^2)
    
    Parameters:
    flat_pos: (N*3,) array of centers
    target_R: target radius
    beta: temperature parameter (higher = sharper)
    """
    pos = flat_pos.reshape(N, 3)
    n = len(pos)
    
    # Precompute indices for upper triangle
    iu_i, iu_j = np.triu_indices(n, k=1)
    
    # 1. Pairwise terms
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    pair_dists = dists[iu_i, iu_j]
    
    # Penalty argument: beta * (2R - d)
    # If d < 2R, arg > 0, softplus is large.
    # If d > 2R, arg < 0, softplus is small.
    overlap_arg = beta * (2.0 * target_R - pair_dists)
    sp_overlap = _softplus(overlap_arg)
    
    energy = np.sum(sp_overlap**2)
    
    grad = np.zeros_like(pos)
    
    # Gradient of sum(sp(x)^2) wrt x is 2 * sp(x) * sigmoid(x)
    # Chain rule: dE/dpos_i = sum_j (dE/darg_ij * darg_ij/dd_ij * dd_ij/dpos_i)
    # dE/darg = 2 * sp * sig
    # darg/dd = -beta
    # dd/dpos_i = (pos_i - pos_j) / d
    
    sig_overlap = 1.0 / (1.0 + np.exp(-overlap_arg))
    
    # Coefficient for each pair
    # coeff = 2 * sp * sig * (-beta) / d
    # Note: dE/dpos_i += coeff * (pos_i - pos_j)
    # dE/dpos_j += -coeff * (pos_i - pos_j)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff = 2.0 * sp_overlap * sig_overlap * (-beta) / pair_dists
        coeff[pair_dists < 1e-12] = 0.0
        
    vecs = pos[iu_i] - pos[iu_j]
    
    # Accumulate gradients
    # grad[i] += coeff * vec
    # grad[j] -= coeff * vec
    np.add.at(grad, iu_i, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, iu_j, -coeff[:, np.newaxis] * vecs)
    
    # 2. Wall terms
    # d_wall = 0.5 - max_k |pos_ik|
    # We use a smooth approximation for max? 
    # Actually, for walls, we can just use the component that is closest to the boundary.
    # To keep it differentiable and simple, we penalize all coordinates that are close to the boundary?
    # Or better: use the same softplus on the actual distance to the wall.
    # d_wall_i = 0.5 - max_k |pos_ik|
    # This max is not differentiable. 
    # Alternative: Penalize 0.5 - |pos_ik| for all k? 
    # If we do that, a sphere in the center is penalized for all 6 walls.
    # This is a valid relaxation. It encourages staying in the center.
    
    abs_pos = np.abs(pos)
    wall_dists = 0.5 - abs_pos
    # wall_dists shape (N, 3)
    
    # Penalty if wall_dist < R
    wall_arg = beta * (target_R - wall_dists)
    sp_wall = _softplus(wall_arg)
    
    energy += np.sum(sp_wall**2)
    
    sig_wall = 1.0 / (1.0 + np.exp(-wall_arg))
    
    # dE/dwall = 2 * sp * sig * (-beta)
    # dwall/dpos = -sign(pos)
    # dE/dpos = dE/dwall * (-sign(pos)) = 2 * beta * sp * sig * sign(pos)
    
    signs = np.sign(pos)
    signs[signs == 0] = 1.0
    
    grad_wall = 2.0 * beta * sp_wall * sig_wall * signs
    grad += grad_wall
    
    return energy, grad.flatten()

def _optimize_soft(flat_pos, target_R, beta, max_iter=100, time_limit=5.0):
    """
    Minimize the soft energy for a given radius and beta.
    """
    def obj_and_grad(x):
        val, g = energy_and_gradient_soft(x, target_R, beta)
        return val, g

    bounds = [(-HALF, HALF)] * (N * 3)
    
    start_time = time.time()
    try:
        res = minimize(
            obj_and_grad,
            flat_pos,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if np.all(np.isfinite(res.x)):
            return res.x
    except Exception:
        pass
    return flat_pos

def _polish_hard_min(pos, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    High-precision polish pass using the hard-min objective.
    """
    def hard_min_obj_and_grad(flat_pos):
        pos = flat_pos.reshape(N, 3)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        d_walls = HALF - np.max(np.abs(pos), axis=1)
        min_wall_val = np.min(d_walls)
        min_wall_idx = np.argmin(d_walls)
        wi = min_wall_idx
        wk = np.argmax(np.abs(pos[wi]))
        
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

    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            hard_min_obj_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: Grid
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
    
    best_pos = make_grid(5, 4, 5)
    best_r = compute_min_dist(best_pos)
    
    rng = np.random.default_rng(42)
    
    # Start with a conservative radius
    current_r = 0.105
    best_pos = np.clip(best_pos, -0.45, 0.45)
    
    # Softplus parameters
    # Start with low beta (soft) to explore, increase to sharpen
    beta_start = 10.0
    beta_end = 100.0
    
    step_size_r = 0.0005
    consecutive_failures = 0
    max_failures = 3
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 20.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size_r *= 0.7
            consecutive_failures = 0
            if step_size_r < 1e-7:
                break
        
        success = False
        best_candidate = best_pos
        best_candidate_r = current_r
        
        # Try a few perturbations
        for _ in range(3):
            if time.time() > deadline - 20.0:
                break
                
            # Perturb current best
            if _ == 0:
                start_pos = best_pos
            else:
                noise_scale = 0.01 * _
                start_pos = best_pos + rng_search.normal(0, noise_scale, size=(N, 3))
                start_pos = np.clip(start_pos, -0.49, 0.49)
            
            # Anneal beta: Start low, end high
            # This helps escape local minima by allowing "tunneling" through low-barrier regions
            beta_curr = beta_start
            steps_beta = 3
            pos_curr = start_pos
            
            for b_step in range(steps_beta):
                # Interpolate beta
                beta_curr = beta_start + (beta_end - beta_start) * (b_step / max(1, steps_beta - 1))
                
                # Optimize for this beta and target radius
                pos_curr = _optimize_soft(pos_curr.flatten(), target_r, beta_curr, max_iter=50, time_limit=5.0)
                pos_curr = pos_curr.reshape(N, 3)
                
                # Check feasibility
                r_check = compute_min_dist(pos_curr)
                
                if r_check > best_candidate_r:
                    best_candidate = pos_curr
                    best_candidate_r = r_check
                    
                # If we achieved the target radius with high beta, we are likely feasible
                if r_check >= target_r - 1e-5 and beta_curr > beta_end * 0.5:
                    success = True
                    break
            
            if success:
                break
        
        if success and best_candidate_r >= target_r - 1e-6:
            best_pos = best_candidate
            best_r = best_candidate_r
            current_r = best_r
            consecutive_failures = 0
        else:
            # If we improved but didn't hit target, take the improvement
            if best_candidate_r > current_r:
                best_pos = best_candidate
                best_r = best_candidate_r
                current_r = best_r
                # Don't count as failure if we improved
                consecutive_failures = 0 
            else:
                consecutive_failures += 1

    # Final Polish with Hard Min
    if time.time() < deadline - 5.0:
        polished_pos = _polish_hard_min(best_pos, max_iter=500)
        final_r = compute_min_dist(polished_pos)
        if final_r > best_r:
            best_r = final_r
            best_pos = polished_pos

    return best_pos
