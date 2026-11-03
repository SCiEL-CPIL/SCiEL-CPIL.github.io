"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius_for_centres(centres):
    """Compute the maximum common radius for given centres (exact)."""
    n = len(centres)
    if n < 2:
        return 0.5 - np.max(np.abs(centres))
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall)

def _soft_min_and_grad(values, beta):
    """
    Compute soft-min of values and its gradient w.r.t. values.
    Returns (soft_min_value, gradient_array).
    """
    beta_safe = max(beta, 1e-6)
    min_val = np.min(values)
    shifted = values - min_val
    exp_vals = np.exp(-beta_safe * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    log_sum = np.log(sum_exp)
    soft_min_val = min_val - log_sum / beta_safe
    grad = -weights
    return soft_min_val, grad

def _smooth_objective_and_grad(centres, beta):
    """
    Smooth surrogate objective: minimize -soft_min of all constraints.
    Returns (value, gradient_vector).
    """
    n = len(centres)
    # Pairwise distances
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    
    # Pair constraints: dists / 2
    pair_vals = dists / 2.0
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_vals[iu]
    
    all_vals = np.concatenate([pair_flat, wall_dists.ravel()])
    
    soft_min_val, weights = _soft_min_and_grad(all_vals, beta)
    
    grad = np.zeros_like(centres)
    
    # Pair contributions
    for idx in range(len(iu[0])):
        i = iu[0][idx]
        j = iu[1][idx]
        w = weights[idx]
        diff = centres[i] - centres[j]
        dist = dists[i, j]
        if dist > 1e-12:
            # d/dcentres[i] of (|c_i - c_j|/2) = (c_i - c_j) / (2*dist)
            # d/dcentres[j] of (|c_i - c_j|/2) = -(c_i - c_j) / (2*dist)
            # But we want grad of -soft_min, so:
            # grad[i] += -w * (c_i - c_j) / (2*dist) ... wait
            # soft_min = min_val - log_sum/beta
            # d(soft_min)/d(values) = -weights
            # d(values_k)/d(centres) for pair k:
            #   for pair (i,j): value = |c_i-c_j|/2
            #   d(value)/d(c_i) = (c_i-c_j)/(2|c_i-c_j|)
            #   d(value)/d(c_j) = -(c_i-c_j)/(2|c_i-c_j|)
            # d(soft_min)/d(c_i) = sum_k d(soft_min)/d(v_k) * d(v_k)/d(c_i)
            #                     = -w_k * (c_i-c_j)/(2*dist)
            # We want to minimize -soft_min, so objective grad = -d(soft_min)/d(c)
            # grad[i] += -(-w_k) * (c_i-c_j)/(2*dist) = w_k * (c_i-c_j)/(2*dist)
            grad[i] += w * diff / (2.0 * dist)
            grad[j] += w * (-diff) / (2.0 * dist)
    
    # Wall contributions
    wall_flat_idx = 0
    for i in range(n):
        for k in range(3):
            w = weights[wall_flat_idx]
            ck = centres[i, k]
            # wall_dist = 0.5 - |c_ik|
            # d(wall_dist)/d(c_ik) = -sign(c_ik)
            # d(soft_min)/d(c_ik) = -w * (-sign(c_ik)) = w * sign(c_ik)
            # objective grad = -d(soft_min)/d(c_ik) = -w * sign(c_ik)
            if ck > 1e-12:
                grad[i, k] += -w
            elif ck < -1e-12:
                grad[i, k] += w
            else:
                grad[i, k] += 0.0
            wall_flat_idx += 1
    
    return -soft_min_val, -grad

def _optimize_continuation(start, deadline, max_beta=300.0, max_iter_per_beta=200):
    """
    Continuation method: ramp up beta from small to large.
    At each beta, run L-BFGS-B to minimize -soft_min.
    """
    centres = start.copy()
    n = len(centres)
    
    # Beta schedule: geometric progression
    betas = [1.0, 3.0, 8.0, 20.0, 50.0, 100.0, 200.0, max_beta]
    
    bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (n * 3)
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def objective_flat(x):
            c = x.reshape(n, 3)
            val, grad = _smooth_objective_and_grad(c, beta)
            return val, grad.ravel()
        
        x0 = centres.ravel()
        
        try:
            result = minimize(
                objective_flat,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 40}
            )
            centres = result.x.reshape(n, 3)
        except Exception:
            pass
        
        centres = np.clip(centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return centres, _radius_for_centres(centres)

def _initial_grid(n_spheres):
    """Create an initial grid configuration."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_spheres]

def _bcc_lattice(n_spheres):
    """Create a BCC-like initial configuration."""
    # Body-centered cubic lattice points in a cube
    # For a 3x3x3 grid of BCC cells, we have 27*2 = 54 points
    # We'll take a subset
    m = 3
    points = []
    spacing = 1.0 / m
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                # Center
                points.append([x + spacing/2, y + spacing/2, z + spacing/2])
    points = np.array(points[:n_spheres])
    return points

def _fcc_lattice(n_spheres):
    """Create an FCC-like initial configuration."""
    # Face-centered cubic lattice
    m = 4  # 4x4x4 grid gives 64 corner + face points, we'll subset
    points = []
    spacing = 1.0 / m
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                # Face centers
                if i < m-1:
                    points.append([x + spacing/2, y, z])
                if j < m-1:
                    points.append([x, y + spacing/2, z])
                if k < m-1:
                    points.append([x, y, z + spacing/2])
    points = np.array(points[:n_spheres])
    return points

def _random_perturb(centres, sigma=0.02):
    """Randomly perturb centres."""
    perturbed = centres + np.random.randn(*centres.shape) * sigma
    return np.clip(perturbed, -0.49, 0.49)

def _dynamics_shake(centres, deadline, n_steps=5000, dt=0.001, damping=0.95, k_repel=1.0, k_wall=5.0):
    """
    Rigid-body dynamics simulation with repulsive forces to shake out local minima.
    Uses velocity-Verlet-like integration with strong damping.
    """
    n = len(centres)
    pos = centres.copy().astype(float)
    vel = np.zeros_like(pos)
    
    best_pos = pos.copy()
    best_r = _radius_for_centres(best_pos)
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Repulsive forces between spheres (inverse distance squared)
        for i in range(n):
            for j in range(i+1, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    # Jitter to avoid singularity
                    diff = np.random.randn(3) * 1e-5
                    dist = np.linalg.norm(diff)
                if dist < 0.5:  # Only interact if somewhat close
                    f = k_repel * diff / (dist**2 + 1e-6)
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion
        for i in range(n):
            for k in range(3):
                if pos[i, k] > 0.3:
                    forces[i, k] -= k_wall * (pos[i, k] - 0.3)
                elif pos[i, k] < -0.3:
                    forces[i, k] -= k_wall * (pos[i, k] + 0.3)
        
        # Update velocity and position (Euler with damping)
        vel = damping * vel + forces * dt
        pos = pos + vel * dt
        
        # Keep inside cube
        pos = np.clip(pos, -0.49, 0.49)
        
        # Check improvement
        r = _radius_for_centres(pos)
        if r > best_r + 1e-12:
            best_r = r
            best_pos = pos.copy()
    
    return best_pos, best_r

def _local_search(centres, deadline, n_iter=1000):
    """
    Local search: iteratively move each sphere to improve the minimum radius.
    """
    best_centres = centres.copy()
    best_r = _radius_for_centres(best_centres)
    n = len(best_centres)
    
    current_centres = best_centres.copy()
    current_r = best_r
    
    step_size = 0.005
    for it in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Pick a random sphere and a random direction
        idx = np.random.randint(n)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Try moving in this direction
        new_centres = current_centres.copy()
        new_centres[idx] += step_size * direction
        new_centres = np.clip(new_centres, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _radius_for_centres(new_centres)
        
        if new_r > current_r + 1e-12:
            current_centres = new_centres
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centres = current_centres.copy()
            step_size = min(step_size * 1.05, 0.02)
        else:
            # Try opposite direction
            new_centres[idx] -= 2 * step_size * direction
            new_centres = np.clip(new_centres, -0.5 + 1e-10, 0.5 - 1e-10)
            new_r = _radius_for_centres(new_centres)
            if new_r > current_r + 1e-12:
                current_centres = new_centres
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centres = current_centres.copy()
                step_size = min(step_size * 1.05, 0.02)
            else:
                step_size = max(step_size * 0.95, 0.001)
    
    return best_centres, best_r

def construct():
    """
    Construct 50 sphere centres in the unit cube [-1/2, 1/2]^3 to maximize
    the common radius.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_grid(N),
        _bcc_lattice(N),
        _fcc_lattice(N),
    ]
    
    best_centres = None
    best_r = -1.0
    
    # Optimize from each initial config
    for init in initial_configs:
        if time.time() > deadline - 30:
            break
        
        opt_centres, opt_r = _optimize_continuation(init, deadline)
        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres
    
    # Dynamics shake from best config
    if time.time() < deadline - 20:
        dyn_centres, dyn_r = _dynamics_shake(best_centres, deadline, n_steps=3000)
        if dyn_r > best_r:
            best_r = dyn_r
            best_centres = dyn_centres
    
    # Re-optimize after dynamics
    if time.time() < deadline - 15:
        opt_centres, opt_r = _optimize_continuation(best_centres, deadline)
        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres
    
    # More restarts with perturbation
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        sigma = 0.02 + 0.01 * restart
        start = _random_perturb(best_centres, sigma=sigma)
        
        opt_centres, opt_r = _optimize_continuation(start, deadline)
        
        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres
    
    # Local search polish
    if time.time() < deadline - 10:
        ls_centres, ls_r = _local_search(best_centres, deadline, n_iter=500)
        if ls_r > best_r:
            best_r = ls_r
            best_centres = ls_centres
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        def objective_flat_high_beta(x):
            c = x.reshape(N, 3)
            val, grad = _smooth_objective_and_grad(c, 500.0)
            return val, grad.ravel()
        
        bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (N * 3)
        x0 = best_centres.ravel()
        
        try:
            result = minimize(
                objective_flat_high_beta,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
            )
            final_centres = result.x.reshape(N, 3)
            final_r = _radius_for_centres(final_centres)
            if final_r > best_r:
                best_r = final_r
                best_centres = final_centres
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
