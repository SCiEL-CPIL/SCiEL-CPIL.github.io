import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = HALF - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        violations = 2.0 * R - dists
        violations[~mask] = 0.0
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        v_ij = violations[i_idx, j_idx]
        d_ij = dists[i_idx, j_idx]
        
        energy = np.sum(v_ij**2)
        
        grad = np.zeros_like(centers)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            v = v_ij[k]
            d = d_ij[k]
            if d < 1e-12:
                continue
            factor = -2.0 * v / d
            vec = (centers[i] - centers[j])
            force = factor * vec
            
            grad[i] += force
            grad[j] -= force
    else:
        energy = 0.0
        grad = np.zeros_like(centers)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = HALF - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists
        violations_wall[~mask_wall] = 0.0
        
        energy += np.sum(violations_wall**2)
        
        for i in range(n):
            if not mask_wall[i]:
                continue
            v = violations_wall[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            force_wall = 2.0 * v * grad_wall_i
            grad[i] += force_wall
            
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=200, time_limit=None):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    Returns the optimized centers.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-HALF, HALF)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def _sgld_step(pos, r_target, n_steps=100, step_size=0.01, noise_scale=0.01, rng=None):
    """
    Stochastic Gradient Langevin Dynamics step.
    Uses the energy gradient as the deterministic force and adds noise for exploration.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    pos = pos.copy()
    bounds = [(-HALF, HALF)] * (N * 3)
    
    for i in range(n_steps):
        energy, grad_flat = _energy_and_gradient(pos.flatten(), r_target)
        grad = grad_flat.reshape(N, 3)
        
        # Anneal noise and step size
        progress = i / n_steps
        current_step = step_size * (1.0 - 0.5 * progress)
        current_noise = noise_scale * (1.0 - 0.8 * progress)
        
        # SGLD update: x <- x + step * grad + noise * sqrt(step) * N(0, I)
        # Note: grad is from minimizing energy, so we move in direction of -grad
        noise = rng.normal(0, 1, size=pos.shape) * current_noise * np.sqrt(current_step)
        pos = pos - current_step * grad + noise
        
        # Project back to bounds (simple clipping for SGLD is often sufficient for local refinement)
        pos = np.clip(pos, -HALF + 1e-6, HALF - 1e-6)
        
        # Early stop if energy is very low (feasible)
        if energy < 1e-12:
            break
            
    return pos

def _fast_repulsion_step(pos, r_target, n_iters=50, lr=0.01, wall_margin=0.001):
    """
    Fast vectorized repulsion-based local search using KD-tree for neighbor queries.
    """
    pos = pos.copy()
    for it in range(n_iters):
        tree = cKDTree(pos)
        
        search_r = 2.0 * r_target + 0.02
        neighbors = tree.query_ball_point(pos, r=search_r, workers=-1)
        
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in neighbors[i]:
                if i == j:
                    continue
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    forces[i] += np.random.normal(0, 1e-4, size=3)
                    continue
                if dist < 2.0 * r_target:
                    strength = (2.0 * r_target - dist) / (2.0 * r_target)
                    forces[i] += strength * d / dist
                else:
                    strength = 0.1 * (search_r - dist) / search_r
                    forces[i] += strength * d / dist
        
        for i in range(N):
            for k in range(3):
                if pos[i, k] > HALF - wall_margin:
                    forces[i, k] -= (pos[i, k] - (HALF - wall_margin)) / wall_margin
                elif pos[i, k] < -HALF + wall_margin:
                    forces[i, k] += ((-HALF + wall_margin) - pos[i, k]) / wall_margin
        
        pos += lr * forces
        pos = np.clip(pos, -HALF + 1e-6, HALF - 1e-6)
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Grid
    # 5x4x5 = 100
    def make_grid(nx, ny, nz):
        xs = np.linspace(-HALF + HALF/nx, HALF - HALF/nx, nx)
        ys = np.linspace(-HALF + HALF/ny, HALF - HALF/ny, ny)
        zs = np.linspace(-HALF + HALF/nz, HALF - HALF/nz, nz)
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)
    
    grid_centers = make_grid(5, 4, 5)
    best_pos = grid_centers.copy()
    best_r = _compute_radius(best_pos)
    
    # List of initializations
    inits = [best_pos]
    # Add random inits
    for _ in range(5):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)
        
    # Strategy: Iterative radius increase with SGLD and L-BFGS-B refinement
    
    # Start with a target radius slightly above current best
    # We will try to increase this target progressively
    
    current_target = best_r * 1.01
    max_target = 0.115 # Known best is ~0.1114, try to go higher
    
    # Main loop
    while time.time() < deadline - 10.0:
        # Try to find a configuration that satisfies current_target
        found = False
        for init in inits:
            if time.time() > deadline - 10.0:
                break
                
            # 1. Try SGLD from init
            sgld_pos = _sgld_step(init, current_target, n_steps=50, step_size=0.005, noise_scale=0.005, rng=rng)
            r_sgld = _compute_radius(sgld_pos)
            
            # 2. Try L-BFGS-B from init (or from sgld_pos if it was better)
            start_pos = sgld_pos if r_sgld > _compute_radius(init) else init
            
            opt_pos, energy = _optimize_for_radius(start_pos, current_target, max_iter=100)
            r_opt = _compute_radius(opt_pos)
            
            # 3. Try L-BFGS-B from init directly
            opt_pos2, energy2 = _optimize_for_radius(init, current_target, max_iter=100)
            r_opt2 = _compute_radius(opt_pos2)
            
            best_candidate_r = max(r_sgld, r_opt, r_opt2)
            
            if best_candidate_r >= current_target:
                # We found a valid configuration for current_target
                # Update best
                if best_candidate_r > best_r:
                    best_r = best_candidate_r
                    if best_candidate_r == r_sgld:
                        best_pos = sgld_pos
                    elif best_candidate_r == r_opt:
                        best_pos = opt_pos
                    else:
                        best_pos = opt_pos2
                    
                    # Update inits to include this new best
                    inits = [best_pos] + inits[:4]
                    
                    # Increase target
                    current_target = best_r * 1.005
                    found = True
                    break
            else:
                # Even if we didn't reach target, maybe we improved best_r
                if best_candidate_r > best_r:
                    best_r = best_candidate_r
                    if best_candidate_r == r_sgld:
                        best_pos = sgld_pos
                    elif best_candidate_r == r_opt:
                        best_pos = opt_pos
                    else:
                        best_pos = opt_pos2
                    inits = [best_pos] + inits[:4]
        
        if not found:
            # If we couldn't increase the target, try a smaller increment or adjust strategy
            # If we are stuck, maybe try a larger perturbation or reset
            # For now, just slightly increase target to keep trying
            current_target = current_target * 1.001
            if current_target > max_target:
                current_target = max_target
        
        # Occasionally add new random inits
        if time.time() % 10.0 < 0.1:
            rc = rng.uniform(-0.4, 0.4, size=(N, 3))
            inits.append(rc)
            if len(inits) > 10:
                inits.pop()

    # Final polish with high precision
    if time.time() < deadline - 5.0:
        # Use a slightly lower target than best_r to ensure convergence
        polish_target = best_r * 0.99
        opt_pos, energy = _optimize_for_radius(best_pos, polish_target, max_iter=500)
        r_opt = _compute_radius(opt_pos)
        if r_opt > best_r:
            best_r = r_opt
            best_pos = opt_pos
            
        # Also try a direct L-BFGS-B on the hard-min objective if available?
        # The energy method is smooth, so it should be fine.
        # Let's do one more pass with a very tight tolerance on the current best
        opt_pos2, energy2 = _optimize_for_radius(best_pos, best_r, max_iter=200)
        r_opt2 = _compute_radius(opt_pos2)
        if r_opt2 > best_r:
            best_r = r_opt2
            best_pos = opt_pos2

    best_pos = np.clip(best_pos, -HALF, HALF)
    return best_pos

# EVOLVE-BLOCK-END
