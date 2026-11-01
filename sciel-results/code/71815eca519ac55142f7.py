"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

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

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.12
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    if len(points) == 0:
        return _get_random_initialization(n, seed=42)
    points = np.array(points)
    if len(points) < n:
        return _get_random_initialization(n, seed=42)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        while True:
            radius = rng.uniform(0, 1.0)
            if radius <= 0.95:
                break
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_ring_initialization(n):
    """Generate ring-based initialization matching the known optimal structure."""
    centers = []
    centers.append([0.0, 0.0])
    # Known packing: 1 center, 6 ring, 12 ring, 12 ring, 9 ring, 10 ring
    rings = [
        (6, 0.2516),
        (12, 0.5032),
        (12, 0.7548),
        (9, 0.8756),
        (10, 0.9242),
    ]
    count = 1
    for count_in_ring, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
    return np.array(centers[:n])

def _objective_soft_min(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints) using log-sum-exp.
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    bound_vals = 1.0 - norms
    
    # All constraint values (we want to maximize the minimum)
    vals = np.concatenate([pair_dists / 2.0, bound_vals])
    m = np.min(vals)
    
    # Log-sum-exp for numerical stability
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = m - (1.0 / beta) * log_sum
    obj = -soft_min
    
    # Gradient computation (vectorized where possible)
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    grad = np.zeros_like(centers)
    
    # Pair gradient contributions
    pair_weights = weights[:n*(n-1)//2]
    # Vectorized: for each pair k, weight w_k, centers[i]-centers[j] direction
    # grad[i] += w_k * (c_i - c_j) / (2 * d_ij)
    # grad[j] -= w_k * (c_i - c_j) / (2 * d_ij)
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-10:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
    
    # Boundary gradient contributions
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm > 1e-10:
            grad[i] -= w * centers[i] / norm
    
    return obj, -grad

def _weighted_subgradient_polish(centers, start_time, max_iters=20000, initial_step=0.003, decay=0.9995):
    """
    Weighted subgradient ascent considering ALL near-tight constraints.
    Uses exponential weighting to focus on tightest constraints while
    also making progress on nearby ones.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    step = initial_step
    n = len(current)
    i_idx, j_idx = np.triu_indices(n, k=1)
    no_improve = 0
    beta = 50.0  # Weighting sharpness
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        # Compute all distances
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        m = np.min(all_vals)
        
        # Exponential weights (soft-min subgradient)
        shifted = all_vals - m
        exp_terms = np.exp(-beta * shifted)
        S = np.sum(exp_terms)
        weights = exp_terms / S
        
        # Compute weighted gradient
        grad = np.zeros_like(current)
        
        # Pair contributions
        pair_weights = weights[:len(i_idx)]
        for k in range(len(pair_weights)):
            w = pair_weights[k]
            if w < 1e-8:
                continue
            i, j = i_idx[k], j_idx[k]
            d = dists[i, j]
            if d > 1e-10:
                vec = (current[i] - current[j]) / d
                grad[i] += w * vec * 0.5
                grad[j] -= w * vec * 0.5
        
        # Boundary contributions
        bound_weights = weights[len(i_idx):]
        for i in range(n):
            w = bound_weights[i]
            if w < 1e-8:
                continue
            norm = norms[i]
            if norm > 1e-10:
                grad[i] -= w * current[i] / norm
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            current += grad / gnorm * step
        
        # Project to feasible region
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.999
        if np.any(mask):
            current[mask] *= 0.999 / new_norms[mask]
        
        # Check improvement
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
            no_improve = 0
        else:
            no_improve += 1
            # Decay step and increase beta for sharper focus
            step *= decay
            if no_improve > 500:
                beta *= 1.5
                step *= 0.8
            if no_improve > 5000:
                break
        
        # Periodically recompute beta based on constraint spread
        if it % 1000 == 0 and it > 0:
            spread = np.max(all_vals) - m
            if spread > 1e-6:
                beta = min(1000.0, 3.0 / spread)
    
    return best_centers, best_score

def _optimize_from_start(centers, start_time):
    """
    Multi-stage optimization:
    1. L-BFGS-B with soft min (beta annealing 30 -> 5000)
    2. Weighted subgradient polish
    """
    current = centers.copy()
    
    # Stage 1: L-BFGS-B with increasing beta
    betas = [30.0, 100.0, 300.0, 1000.0, 3000.0, 5000.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 8:
            break
        try:
            maxiter = 200 if beta < 500 else 100
            res = minimize(
                _objective_soft_min, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    # Verify it's valid
                    norms = np.linalg.norm(candidate, axis=1)
                    if np.all(norms < 1.0):
                        current = candidate
        except Exception:
            pass
            
    # Stage 2: Weighted Sub-gradient Polish
    if time.time() - start_time < DEADLINE - 3:
        current, _ = _weighted_subgradient_polish(current, start_time, max_iters=20000, initial_step=0.003)
        
    return current

def _refine_with_high_beta(centers, start_time, max_betas=6):
    """
    Refine using very high beta soft-min to sharpen the minimum.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    betas = [5000, 10000, 20000, 50000, 100000, 200000][:max_betas]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 3:
            break
        try:
            res = minimize(
                _objective_soft_min, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 80, 'gtol': 1e-9}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    norms = np.linalg.norm(candidate, axis=1)
                    if np.all(norms < 1.0):
                        score = _compute_score(candidate)
                        if score > best_score:
                            best_score = score
                            best_centers = candidate.copy()
                            current = candidate
        except Exception:
            pass
            
    return best_centers, best_score

def _perturb_and_retry(centers, start_time, n_perturbations=5):
    """
    Perturb the best solution slightly and re-optimize to escape local minima.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    rng = np.random.default_rng(12345)
    
    for p in range(n_perturbations):
        if time.time() - start_time > DEADLINE - 8:
            break
        
        # Create a perturbed copy with decreasing noise
        perturbed = best_centers.copy()
        noise_scale = 0.008 * (1.0 + p * 0.3)
        noise = rng.normal(0, noise_scale, size=perturbed.shape)
        perturbed += noise
        
        # Ensure validity
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            perturbed[mask] *= 0.95 / norms[mask]
            
        # Optimize the perturbed version
        optimized = _optimize_from_start(perturbed, start_time)
        if np.all(np.isfinite(optimized)):
            score = _compute_score(optimized)
            if score > best_score:
                best_score = score
                best_centers = optimized.copy()
                
    return best_centers, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations - focus on structured ones that match known packing
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=777),
    ]
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # L-BFGS-B + subgradient optimization
        if time.time() - start_time < DEADLINE - 8:
            optimized = _optimize_from_start(init, start_time)
            if np.all(np.isfinite(optimized)):
                r = _compute_score(optimized)
                if r > best_score:
                    best_score = r
                    best_centers = optimized.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 3:
        best_centers, best_score = _weighted_subgradient_polish(best_centers, start_time, max_iters=30000, initial_step=0.001, decay=0.9998)
        
    # High-beta refinement to sharpen the minimum
    if time.time() - start_time < DEADLINE - 3:
        refined, refined_score = _refine_with_high_beta(best_centers, start_time, max_betas=6)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined
        
    # Perturbation-retry loop to escape local minima
    if time.time() - start_time < DEADLINE - 8:
        perturbed, perturbed_score = _perturb_and_retry(best_centers, start_time, n_perturbations=4)
        if perturbed_score > best_score:
            best_score = perturbed_score
            best_centers = perturbed
            # Polish the new best
            if time.time() - start_time < DEADLINE - 3:
                best_centers, best_score = _weighted_subgradient_polish(best_centers, start_time, max_iters=15000, initial_step=0.001)
        
    # Final ultra-precise polish
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _weighted_subgradient_polish(best_centers, start_time, max_iters=20000, initial_step=0.0005, decay=0.9999)
        
    return best_centers
# EVOLVE-BLOCK-END
