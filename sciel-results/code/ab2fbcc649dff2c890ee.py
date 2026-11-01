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

def _get_ring_initialization(n, num_rings=3):
    """Generate ring-based initialization."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = 0.3 + ring_idx * 0.25
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
        radius = rng.uniform(0, 0.85)
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
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    obj = -_soft_min(vals, beta)
    
    exp_vals = np.exp(-beta * (vals - np.min(vals)))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    pair_weights = weights[:n*(n-1)//2]
    # Vectorized gradient computation for pairs
    vecs = centers[i_idx] - centers[j_idx]
    dists_safe = np.maximum(dists[i_idx, j_idx], 1e-8)
    unit_vecs = vecs / dists_safe[:, np.newaxis]
    
    grad_pairs = (pair_weights[:, np.newaxis] * unit_vecs * 0.5)
    np.add.at(grad, i_idx, grad_pairs)
    np.add.at(grad, j_idx, -grad_pairs)
    
    bound_weights = weights[n*(n-1)//2:]
    norms_safe = np.maximum(norms, 1e-8)
    radial_vecs = centers / norms_safe[:, np.newaxis]
    grad[n*(n-1)//2:] += 0  # no-op, just to keep indexing clear
    grad -= (bound_weights[:, np.newaxis] * radial_vecs)
    
    return obj, -grad

def _feasible_radius(centers, r):
    """Check if all circles of radius r fit without overlap and inside unit circle."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min_pair >= 2 * r and min_bound >= r

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

def _binary_search_radius(centers, start_time, r_lo=0.1, r_hi=0.15, tol=1e-6, max_iter=30):
    """Binary search on radius with gradient-based position optimization."""
    current = centers.copy()
    best_r = r_lo
    best_centers = current.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to find a feasible configuration for radius r_mid
        # Use soft-min optimization to maximize the minimum constraint
        optimized = _optimize_from_start(current, beta=500.0, maxiter=60, gtol=1e-8)
        if not np.all(np.isfinite(optimized)):
            r_hi = r_mid
            continue
        
        # Check feasibility
        score = _compute_score(optimized)
        if score >= r_mid:
            r_lo = r_mid
            best_r = r_mid
            best_centers = optimized.copy()
        else:
            r_hi = r_mid
            current = optimized  # Use the optimized position for next attempt
        
        if r_hi - r_lo < tol:
            break
    
    return best_centers, best_r

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=60, gtol=1e-8)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

def _local_search(centers, start_time, max_iters=30, step_size=0.02):
    """Simple local search with random perturbations."""
    current = centers.copy()
    current_score = _compute_score(current)
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Try random perturbation
        noisy = current + rng.normal(0, step_size, current.shape)
        
        # Keep inside container
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            noisy[mask] *= 0.98 / norms[mask]
        
        # Optimize the perturbed solution
        optimized = _optimize_from_start(noisy, beta=200.0, maxiter=50, gtol=1e-8)
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
    for beta in [2000.0, 5000.0, 10000.0]:
        if time.time() - start_time > DEADLINE - 5:
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
        
        optimized = _optimize_from_start(init, beta=100.0, maxiter=80, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    # Second pass: polish the best solution
    if best_centers is not None and time.time() - start_time < DEADLINE - 30:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Third pass: binary search on radius for precise convergence
    if best_centers is not None and time.time() - start_time < DEADLINE - 20:
        best_centers, best_score = _binary_search_radius(best_centers, start_time, r_lo=best_score * 0.95, r_hi=best_score * 1.05)
    
    # Fourth pass: local search with perturbations
    if best_centers is not None and time.time() - start_time < DEADLINE - 15:
        best_centers, best_score = _local_search(best_centers, start_time, max_iters=15, step_size=0.01)
    
    # Fifth pass: final high-precision polish
    if best_centers is not None and time.time() - start_time < DEADLINE - 5:
        best_centers, best_score = _final_high_precision_polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
# EVOLVE-BLOCK-END
