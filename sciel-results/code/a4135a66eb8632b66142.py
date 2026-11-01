"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_neg(params, beta=50.0):
    """
    Smooth surrogate for the negative radius using soft-min (log-sum-exp).
    We want to maximize min(pairwise_dists/2, boundary_dists).
    Returns negative soft-min so we minimize it.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_vals, boundary_dists.flatten()])
    
    # Soft-min: log-sum-exp(-beta * x) / beta approximates -min(x) for large beta
    # We want to maximize the soft-min, so minimize negative soft-min
    max_val = np.max(all_vals)
    # Numerically stable log-sum-exp
    shifted = -beta * (all_vals - max_val)
    lse = np.log(np.sum(np.exp(shifted))) + beta * max_val
    soft_min_val = lse / beta
    
    return -soft_min_val


def _soft_min_neg_grad(params, beta=50.0):
    """
    Gradient of the soft-min negative objective.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_vals, boundary_dists.flatten()])
    
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Gradient of soft-min w.r.t. all_vals is -weights
    # We need chain rule through pair_vals and boundary_dists
    
    grad = np.zeros_like(params)
    
    # Boundary part
    b_start = len(pair_vals)
    b_weights = weights[b_start:]
    # d(boundary_dists)/dc = -sign(c)
    b_grad = -np.sign(c) * b_weights.reshape(n, 2)
    grad += b_grad.flatten()
    
    # Pair part
    p_weights = weights[:len(pair_vals)]
    # Need to compute gradient of pdist
    # For each pair (i,j) with i<j, the contribution is p_weights[k] * d(pair_vals[k])/dc
    # pair_vals[k] = dist(i,j)/2
    # d(dist(i,j))/dc_i = (c_i - c_j) / dist(i,j)
    # d(dist(i,j))/dc_j = (c_j - c_i) / dist(i,j)
    
    # Compute pairwise distances again for gradient
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_matrix = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists_matrix, 1.0)
    
    # Only upper triangle
    for k in range(len(p_weights)):
        # Convert linear index to (i,j)
        i = int((np.sqrt(8*k + 1) - 1) / 2)
        j = i + 1 + int(k - i*(i+1)/2)
        if j >= n:
            continue
        w = p_weights[k]
        d = dists_matrix[i, j]
        if d < 1e-15:
            continue
        diff = (c[i] - c[j]) / d
        grad[2*i:2*i+2] += w * diff
        grad[2*j:2*j+2] += w * (-diff)
    
    return grad


def _hex_init():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.085, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init(seed=0):
    """Random initialization with some structure."""
    np.random.seed(seed)
    c = np.random.uniform(-0.48, 0.48, size=(N, 2))
    return c


def _force_directed(c0, n_iter=200, k=0.008):
    """
    Force-directed relaxation: circles repel each other.
    Uses vectorized operations for speed.
    """
    c = c0.copy()
    n = c.shape[0]
    
    for it in range(n_iter):
        # Compute all pairwise differences and distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive forces: F = k / dist^2 * direction
        # Only consider pairs within a cutoff
        cutoff = 0.2
        mask = dists < cutoff
        np.fill_diagonal(mask, False)
        
        # Compute forces
        forces = np.zeros_like(c)
        safe_dists = np.where(dists > 1e-10, dists, 1e10)
        f_magnitude = np.where(mask, k / (safe_dists**2), 0.0)
        
        # Force direction is along the difference vector
        force_vec = f_magnitude[:, :, np.newaxis] * diffs / safe_dists[:, :, np.newaxis]
        
        # Sum forces for each particle
        forces = np.sum(force_vec, axis=1)
        
        # Weak attraction to center to prevent drift
        forces -= 0.0005 * c
        
        # Damping
        damping = 0.8 * (1.0 - it / n_iter) + 0.2
        step = damping * 0.01
        c += step * forces
        
        # Clip to bounds
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c


def _velocity_verlet(c0, n_iter=1000, dt=0.002, damping=0.98):
    """
    Velocity-Verlet integrator with repulsive potential.
    """
    c = c0.copy()
    v = np.zeros_like(c)
    n = c.shape[0]
    
    def _potential_and_force(pos):
        diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        # Repulsive potential: U = k / dist (softened)
        k = 0.01
        cutoff = 0.15
        mask = dists < cutoff
        np.fill_diagonal(mask, False)
        
        # Force = -dU/dc = k / dist^2 * direction
        safe_dists = np.where(dists > 1e-10, dists, 1e10)
        f_magnitude = np.where(mask, k / (safe_dists**2), 0.0)
        force_vec = f_magnitude[:, :, np.newaxis] * diffs / safe_dists[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Weak centering force
        forces -= 0.0003 * pos
        
        # Potential energy
        U = np.sum(np.where(mask, k / safe_dists, 0.0))
        
        return U, forces
    
    U, F = _potential_and_force(c)
    
    for it in range(n_iter):
        # Velocity Verlet
        c += v * dt + 0.5 * F * dt**2
        v_new, F_new = _potential_and_force(c)
        v = (v + 0.5 * (F + F_new) * dt) * damping
        v = v_new  # Actually, let's use the new velocity
        
        # Recompute properly
        c_half = c0 if it == 0 else c
        # Standard Verlet:
        # v_half = v + 0.5 * F * dt
        # c_new = c + v_half * dt
        # F_new = force(c_new)
        # v_new = v_half + 0.5 * F_new * dt
        # v = v_new * damping
        
        # Clip to bounds
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c


def _optimize_lbfgs(c0, max_time=20.0, beta_start=20.0, beta_end=200.0):
    """
    Local optimization using L-BFGS-B with a smooth surrogate objective.
    Uses annealing of beta to gradually sharpen the soft-min toward the true min.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Smooth optimization with increasing beta
    betas = [beta_start, 50.0, 100.0, beta_end]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        def _obj(params):
            return _soft_min_neg(params, beta=beta)
        
        def _grad(params):
            return _soft_min_neg_grad(params, beta=beta)
        
        try:
            result = minimize(
                _obj,
                _grad,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Phase 2: High-precision polish with exact objective
    if time.time() < start_time + max_time - 2:
        def _exact_obj(params):
            c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r = _compute_radius(c)
            return -r
        
        try:
            result = minimize(
                _exact_obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _active_set_polish(c0, max_time=10.0):
    """
    Polish by identifying the active constraints (tightest pairs and boundaries)
    and doing a focused optimization.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    
    # Identify active constraints
    n = c.shape[0]
    dists = pdist(c, metric='euclidean')
    boundary_dists = 0.5 - np.abs(c)
    
    min_pair_dist = np.min(dists)
    min_boundary = np.min(boundary_dists)
    r = min(min_pair_dist / 2.0, min_boundary)
    
    # Find pairs close to the minimum
    pair_idx = np.argwhere(pdist(c, metric='euclidean') < r * 2.0 * 1.05)
    
    # Do a few iterations of local optimization focusing on these
    for _ in range(3):
        if time.time() > start_time + max_time:
            break
        
        def _obj(params):
            c_local = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            return -_compute_radius(c_local)
        
        try:
            result = minimize(
                _obj,
                c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > r:
                c = c_new
                r = r_new
        except Exception:
            break
    
    return c, r


def _high_precision_polish(c0, max_time=5.0):
    """Final high-precision polish pass."""
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    
    def _obj(params):
        c_local = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        return -_compute_radius(c_local)
    
    try:
        result = minimize(
            _obj,
            c.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > r:
            c = c_new
            r = r_new
    except Exception:
        pass
    
    return c, r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # Initialize with hexagonal pattern
    c_hex = _hex_init()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Optimize hexagonal initialization
    if time.time() < deadline - 15:
        c_opt, r_opt = _optimize_lbfgs(c_hex, max_time=min(35.0, deadline - time.time() - 10))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try multiple random initializations with force-directed relaxation
    for seed in [123, 456, 789, 101, 202]:
        if time.time() > deadline - 30:
            break
        
        c_rand = _random_init(seed)
        
        # Apply force-directed relaxation
        c_fd = _force_directed(c_rand, n_iter=150, k=0.008)
        r_fd = _compute_radius(c_fd)
        
        if r_fd > best_radius:
            best_radius = r_fd
            best_c = c_fd.copy()
        
        # Also try velocity-Verlet
        if time.time() < deadline - 25:
            c_vv = _velocity_verlet(c_rand, n_iter=500, dt=0.001, damping=0.98)
            r_vv = _compute_radius(c_vv)
            if r_vv > best_radius:
                best_radius = r_vv
                best_c = c_vv.copy()
        
        # Optimize the best of force-directed and velocity-Verlet
        if time.time() < deadline - 20:
            c_opt, r_opt = _optimize_lbfgs(c_fd, max_time=min(20.0, deadline - time.time() - 5))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Active set polish on the best solution
    if best_c is not None and time.time() < deadline - 10:
        c_polish, r_polish = _active_set_polish(best_c, max_time=min(8.0, deadline - time.time() - 5))
        if r_polish > best_radius:
            best_radius = r_polish
            best_c = c_polish.copy()
    
    # Final high-precision polish
    if best_c is not None and time.time() < deadline - 3:
        c_final, r_final = _high_precision_polish(best_c, max_time=min(5.0, deadline - time.time() - 1))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final.copy()
    
    if best_c is None:
        best_c = _hex_init()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
