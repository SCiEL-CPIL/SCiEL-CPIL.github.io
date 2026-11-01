"""
Packing 50 equal circles in a unit circle using L-BFGS-B with Simulated Annealing.
Refined with fixed broadcasting, better initialization, and staged optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill

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
    
    # Select points closest to the center to form a dense core
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, num_rings=5):
    """Generate ring-based initialization with better spacing."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    # Distribute points across rings with increasing radius
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
        # Fixed broadcasting: (K, 1) * (K, 2)
        centers[mask] *= (max_radius / norms[mask])[:, np.newaxis]
    return centers

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints) using numerically stable log-sum-exp
    params: (n, 2) array of centers
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    # Numerically stable soft-min: -log(sum(exp(-beta * vals))) / beta
    # = -(min(vals) + log(sum(exp(-beta * (vals - min(vals))))) / beta
    m = np.min(vals)
    shifted = vals - m
    # Clamp shifted values to avoid underflow/overflow in exp
    shifted_clamped = np.clip(shifted, 0, 500) 
    log_sum = np.log(np.sum(np.exp(-beta * shifted_clamped)))
    
    obj = -(m - log_sum / beta)
    
    # Weights: w_i = exp(-beta * (vals_i - m)) / sum(exp(-beta * (vals_j - m)))
    exp_shifted = np.exp(-beta * shifted_clamped)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        sum_exp = 1e-300
    weights = exp_shifted / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    n_pairs = n*(n-1)//2
    pair_weights = weights[:n_pairs]
    d_vecs = centers[i_idx] - centers[j_idx]
    d_mags = pair_dists
    safe_mags = np.where(d_mags > 1e-10, d_mags, 1.0)
    unit_vecs = d_vecs / safe_mags[:, np.newaxis]
    
    # For each pair k: grad[i] += w_k * unit_vec[k] / 2, grad[j] -= w_k * unit_vec[k] / 2
    # Vectorize using np.add.at
    half_w = (pair_weights / 2.0)[:, np.newaxis]
    np.add.at(grad, i_idx, half_w * unit_vecs)
    np.add.at(grad, j_idx, -half_w * unit_vecs)
    
    # Boundary contributions
    bound_weights = weights[n_pairs:]
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    # grad -= (bound_weights / safe_norms)[:, np.newaxis] * centers
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

def _nelder_mead_polish(centers, maxiter=300):
    """Polish using Nelder-Mead on the raw (nonsmooth) objective."""
    def raw_obj(params):
        c = params.reshape(N, 2)
        # Add a small penalty for points outside to help NM stay in bounds if it drifts
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

def _simulated_annealing(centers, start_time, max_iter=50):
    """
    Simulated annealing with L-BFGS-B steps to escape local minima.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.95
    rng = np.random.default_rng(12345)
    
    for it in range(max_iter):
        if time.time() - start_time > _DEADLINE - 25:
            break
        
        # Adaptive perturbation
        progress = it / max(max_iter, 1)
        noise_scale = 0.01 * (1 - progress) + 0.002
        
        noisy = current + rng.normal(0, noise_scale, current.shape)
        noisy = _project_to_disk(noisy, 0.97)
        
        # Gradient step
        optimized = _optimize_from_start(noisy, beta=200.0, maxiter=50, gtol=1e-8)
        if not np.all(np.isfinite(optimized)):
            continue
        optimized = _project_to_disk(optimized)
        
        new_score = _compute_score(optimized)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, T_min)):
            current = optimized
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best, best_score

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
    
    # Ensure all initializations are valid
    for i in range(len(initializations)):
        initializations[i] = _project_to_disk(initializations[i], 0.95)
        
    # First pass: optimize each initialization
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 60:
            break
        
        # Quick initial optimization with lower beta to find a basin
        optimized = _optimize_from_start(init, beta=100.0, maxiter=50, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            optimized = _project_to_disk(optimized)
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Second pass: Simulated Annealing to escape local minima
    if time.time() - start_time < _DEADLINE - 40:
        sa_centers, sa_score = _simulated_annealing(best_centers, start_time, max_iter=30)
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers
            
    # Third pass: Final Polish
    if time.time() - start_time < _DEADLINE - 10:
        best_centers, best_score = _polish(best_centers, start_time)
        
    # Final check and projection
    best_centers = _project_to_disk(best_centers)
    final_score = _compute_score(best_centers)
    
    return best_centers

# EVOLVE-BLOCK-END
