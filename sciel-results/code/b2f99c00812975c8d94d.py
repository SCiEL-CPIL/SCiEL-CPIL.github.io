"""
Packing 50 equal circles in a unit circle.
Combines soft-min surrogate gradient optimization (current) with rigid-body dynamics
(velocity-Verlet with damping) for escaping local minima, plus binary-search radius
refinement and high-precision polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 142.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return min(1.0 - np.linalg.norm(centers[0]), 0.5)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2.0 * r_est
    sqrt3 = np.sqrt(3.0)
    
    points = []
    max_range = int(2.0 / s) + 5
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) <= 0.98:
                points.append([x, y])
                
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, num_rings=5):
    """Generate ring-based initialization with better spacing."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    ring_radii = np.linspace(0.25, 0.9, num_rings)
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = ring_radii[ring_idx]
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + ring_idx * 0.3
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.88)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _project_to_disk(centers, max_radius=0.99):
    """Project centers to be strictly inside the unit disk."""
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > max_radius
    if np.any(mask):
        centers[mask] *= (max_radius / norms[mask])[:, np.newaxis]
    return centers

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints) using numerically stable log-sum-exp.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    m = np.min(vals)
    shifted = vals - m
    shifted_clamped = np.clip(shifted, 0, 500)
    log_sum = np.log(np.sum(np.exp(-beta * shifted_clamped)))
    
    obj = -(m - log_sum / beta)
    
    exp_shifted = np.exp(-beta * shifted_clamped)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        sum_exp = 1e-300
    weights = exp_shifted / sum_exp
    
    grad = np.zeros_like(centers)
    
    n_pairs = n*(n-1)//2
    pair_weights = weights[:n_pairs]
    d_vecs = centers[i_idx] - centers[j_idx]
    d_mags = pair_dists
    safe_mags = np.where(d_mags > 1e-10, d_mags, 1.0)
    unit_vecs = d_vecs / safe_mags[:, np.newaxis]
    
    half_w = (pair_weights / 2.0)[:, np.newaxis]
    np.add.at(grad, i_idx, half_w * unit_vecs)
    np.add.at(grad, j_idx, -half_w * unit_vecs)
    
    bound_weights = weights[n_pairs:]
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    bound_grad = (bound_weights / safe_norms)[:, np.newaxis] * centers
    grad -= bound_grad
    
    return obj, -grad

def _optimize_from_start(centers, beta, maxiter=100, gtol=1e-8):
    """Optimize from a given starting point using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50}
        )
        if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
            return res.x.reshape(N, 2)
        return centers
    except Exception:
        return centers

def _energy_and_grad(c_flat, r):
    """
    Penalty energy for fixed radius r.
    Energy = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
    """
    c = c_flat.reshape(N, 2)
    energy = 0.0
    grad = np.zeros_like(c)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    overlap = np.maximum(0, 2 * r - dists)
    pair_energy = 0.5 * np.sum(overlap ** 2)
    energy += pair_energy
    
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    np.fill_diagonal(overlap, 0)
    weights = -2.0 * overlap / safe_dists
    np.fill_diagonal(weights, 0)
    
    grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pairs
    
    radii = np.linalg.norm(c, axis=1)
    bound_violation = np.maximum(0, radii - (1.0 - r))
    energy += np.sum(bound_violation ** 2)
    
    safe_radii = np.where(radii > 1e-10, radii, 1.0)
    grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
    grad += grad_bound
    
    return energy, grad

def _solve_for_radius(r, c_init, maxiter=300):
    """Minimize energy for a fixed radius r starting from c_init."""
    c_flat = c_init.flatten()
    
    def obj(x):
        e, _ = _energy_and_grad(x, r)
        return e
        
    def jac(x):
        _, g = _energy_and_grad(x, r)
        return g.flatten()
        
    try:
        res = minimize(obj, c_flat, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        return res.x.reshape(N, 2), res.fun
    except Exception:
        return c_init, np.inf

def _rigid_body_dynamics(centers, target_r, n_steps=500, start_time=None, deadline=None):
    """
    Velocity-Verlet integrator with adaptive damping to find a jammed configuration
    for circles of radius target_r.
    """
    c = centers.copy()
    v = np.zeros_like(c)
    dt = 0.005
    damping = 0.995
    
    def compute_forces(pos):
        """Compute repulsive forces and confining potential."""
        forces = np.zeros_like(pos)
        n = len(pos)
        
        # Pairwise repulsion: F = k * max(0, 2r - d) * unit_vec
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        overlap = np.maximum(0, 2 * target_r - dists)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(safe_dists, 1.0)
        
        # Force magnitude: proportional to overlap
        force_mags = 10.0 * overlap / safe_dists
        np.fill_diagonal(force_mags, 0)
        
        # Sum forces
        forces += np.sum(force_mags[:, :, np.newaxis] * diff, axis=1)
        
        # Confining potential: harmonic wall
        radii = np.linalg.norm(pos, axis=1)
        wall_violation = np.maximum(0, radii - (1.0 - target_r))
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        forces -= 5.0 * wall_violation[:, np.newaxis] * pos / safe_radii[:, np.newaxis]
        
        return forces
    
    f = compute_forces(c)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Velocity-Verlet
        c += v * dt + 0.5 * f * dt * dt
        v_new = v + f * dt
        
        # Damping
        v_new *= damping
        
        f_new = compute_forces(c)
        v = 0.5 * (v_new + f_new * dt)
        
        # Project to disk to keep valid
        c = _project_to_disk(c, 0.995)
        
        # Adaptive damping: increase damping as we approach convergence
        if step > n_steps * 0.7:
            damping = 0.98
    
    return c

def _nelder_mead_polish(centers, maxiter=300):
    """Polish using Nelder-Mead on the raw (nonsmooth) objective."""
    def raw_obj(params):
        c = params.reshape(N, 2)
        c = _project_to_disk(c)
        return -_compute_score(c)
    
    try:
        res = minimize(
            raw_obj,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    # Stage 1: L-BFGS with moderate to high beta
    betas = [500.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=100, gtol=1e-10)
        if np.all(np.isfinite(polished)):
            polished = _project_to_disk(polished)
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    # Stage 2: Nelder-Mead polish on the nonsmooth objective
    if time.time() - start_time < _DEADLINE - 15:
        nm_result = _nelder_mead_polish(current, maxiter=400)
        if np.all(np.isfinite(nm_result)):
            nm_result = _project_to_disk(nm_result)
            r = _compute_score(nm_result)
            if r > best_score:
                best_score = r
                current = nm_result
    
    return current, best_score

def _binary_search_refine(centers, start_time):
    """
    Use binary search on radius with energy minimization to refine the solution.
    """
    current = centers.copy()
    current_r = _compute_score(current)
    
    if time.time() - start_time > _DEADLINE - 30:
        return current, current_r
    
    r_low = current_r
    r_high = min(current_r + 0.005, 0.135)
    
    best_centers = current
    best_r = current_r
    
    # Binary search iterations
    for _ in range(8):
        if time.time() - start_time > _DEADLINE - 25:
            break
        
        r_mid = (r_low + r_high) / 2
        c_opt, e_opt = _solve_for_radius(r_mid, best_centers, maxiter=200)
        
        if e_opt < 1e-8:
            # Feasible
            r_low = r_mid
            best_r = r_mid
            best_centers = c_opt
        else:
            r_high = r_mid
    
    # Final polish at best_r
    if time.time() - start_time < _DEADLINE - 15:
        c_final, e_final = _solve_for_radius(best_r, best_centers, maxiter=300)
        if e_final < 1e-8:
            best_centers = c_final
    
    final_r = _compute_score(best_centers)
    return best_centers, final_r

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
    ]
    
    for i in range(len(initializations)):
        initializations[i] = _project_to_disk(initializations[i], 0.95)
        
    # First pass: optimize each initialization with soft-min surrogate
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 60:
            break
        
        optimized = _optimize_from_start(init, beta=100.0, maxiter=50, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            optimized = _project_to_disk(optimized)
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Second pass: Rigid-body dynamics to escape local minima
    if time.time() - start_time < _DEADLINE - 40:
        target_r = best_score * 1.01
        dyn_centers = _rigid_body_dynamics(best_centers, target_r, n_steps=300, 
                                            start_time=start_time, deadline=_DEADLINE - 30)
        if np.all(np.isfinite(dyn_centers)):
            dyn_centers = _project_to_disk(dyn_centers)
            # Polish after dynamics
            dyn_polished, dyn_score = _polish(dyn_centers, start_time)
            if dyn_score > best_score:
                best_score = dyn_score
                best_centers = dyn_polished
    
    # Third pass: Binary search refinement
    if time.time() - start_time < _DEADLINE - 20:
        bs_centers, bs_score = _binary_search_refine(best_centers, start_time)
        if bs_score > best_score:
            best_score = bs_score
            best_centers = bs_centers
    
    # Fourth pass: Final high-precision polish
    if time.time() - start_time < _DEADLINE - 10:
        best_centers, best_score = _polish(best_centers, start_time)
        
    # Final check and projection
    best_centers = _project_to_disk(best_centers)
    final_score = _compute_score(best_centers)
    
    return best_centers

# EVOLVE-BLOCK-END
