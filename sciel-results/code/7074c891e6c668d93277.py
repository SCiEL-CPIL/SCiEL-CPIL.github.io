"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE = 140.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 4
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
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

def _get_ring_initialization(n):
    """Generate ring-based initialization matching known optimal structure."""
    centers = []
    centers.append([0.0, 0.0])
    
    # Ring structure: 1 + 6 + 12 + 18 + 13 = 50
    ring_configs = [(6, 0.25), (12, 0.45), (18, 0.65), (13, 0.82)]
    
    for count, radius in ring_configs:
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.9)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values."""
    v = np.array(values)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    """
    centers = params.reshape(n, 2)
    
    # Compute pairwise distances vectorized
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    # Soft min
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    S = np.sum(exp_terms)
    weights = exp_terms / S
    obj = -(m - (1.0 / beta) * np.log(S))
    
    # Vectorized gradient computation
    grad = np.zeros_like(centers)
    
    # Pair constraints gradient
    pair_weights = weights[:n*(n-1)//2]
    mask = pair_weights > 1e-12
    if np.any(mask):
        idx = np.where(mask)[0]
        ii = i_idx[idx]
        jj = j_idx[idx]
        dd = pair_dists[idx]
        valid = dd > 1e-8
        if np.any(valid):
            ii_v = ii[valid]
            jj_v = jj[valid]
            dd_v = dd[valid]
            ww_v = pair_weights[idx[valid]]
            
            vecs = (centers[ii_v] - centers[jj_v]) / dd_v[:, np.newaxis]
            # grad[i] += w * vec / 2
            # grad[j] -= w * vec / 2
            np.add.at(grad, ii_v, ww_v[:, np.newaxis] * vecs * 0.5)
            np.add.at(grad, jj_v, -ww_v[:, np.newaxis] * vecs * 0.5)
    
    # Boundary constraints gradient
    bound_weights = weights[n*(n-1)//2:]
    mask_b = bound_weights > 1e-12
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        valid_b = norms[idx_b] > 1e-8
        if np.any(valid_b):
            idx_v = idx_b[valid_b]
            w_v = bound_weights[idx_v]
            norm_v = norms[idx_v]
            # grad[i] -= w * centers[i] / norms[i]
            grad[idx_v] -= (w_v / norm_v)[:, np.newaxis] * centers[idx_v]
    
    return obj, -grad

def _optimize_from_start(centers, beta, maxiter=80, gtol=1e-6):
    """Optimize from a given starting point."""
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=60, gtol=1e-8)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

def _local_search(centers, start_time, max_iters=25, step_size=0.02):
    """Simple local search with random perturbations."""
    current = centers.copy()
    current_score = _compute_score(current)
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # Try random perturbation
        noisy = current + rng.normal(0, step_size, current.shape)
        
        # Keep inside container
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            noisy = noisy.copy()
            noisy[mask] *= 0.98 / norms[mask]
        
        # Optimize the perturbed solution
        optimized = _optimize_from_start(noisy, beta=100.0, maxiter=50, gtol=1e-8)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > current_score:
                current_score = r
                current = optimized.copy()
    
    return current, current_score

def _final_high_precision_polish(centers, start_time):
    """Final high-precision polish with very high beta and tight tolerances."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    # Very high beta for near-exact min
    for beta in [2000.0, 4000.0, 8000.0]:
        if time.time() - start_time > DEADLINE - 3:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=150, gtol=1e-10)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

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
        _get_random_initialization(N, seed=999),
        _get_random_initialization(N, seed=555),
        _get_random_initialization(N, seed=777),
    ]
    
    # First pass: optimize each initialization with moderate beta
    for init in initializations:
        if time.time() - start_time > DEADLINE - 50:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
        
        optimized = _optimize_from_start(init, beta=50.0, maxiter=80, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    # Second pass: polish the best solution
    if best_centers is not None and time.time() - start_time < DEADLINE - 30:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Third pass: local search with perturbations
    if best_centers is not None and time.time() - start_time < DEADLINE - 20:
        best_centers, best_score = _local_search(best_centers, start_time, max_iters=20, step_size=0.015)
    
    # Fourth pass: final high-precision polish
    if best_centers is not None and time.time() - start_time < DEADLINE - 3:
        best_centers, best_score = _final_high_precision_polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
# EVOLVE-BLOCK-END
