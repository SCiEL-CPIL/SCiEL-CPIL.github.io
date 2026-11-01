# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    n = len(centers)
    if n != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to boundary
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    score = min(min_pair / 2.0, min_bound)
    return score if score > 0 else 0.0

def _soft_min(values, beta=50.0):
    """
    Compute the soft minimum of an array of values.
    Uses log-sum-exp for numerical stability.
    """
    v = np.asarray(values)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_grad(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    Returns: (objective_value, gradient)
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraints: we want to maximize the minimum of these
    # Constraints: d_pair/2 and d_bound
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # Soft min
    obj = -_soft_min(vals, beta)
    
    # Gradient calculation
    v_min = np.min(vals)
    exp_vals = np.exp(-beta * (vals - v_min))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
            
    # Boundary part
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        if norms[i] > 1e-8:
            grad[i] -= w * centers[i] / norms[i]
            
    return obj, grad

def _get_ring_initialization(n, rng):
    """
    Generate an initialization with points distributed in rings.
    """
    centers = np.zeros((n, 2))
    # Center point
    centers[0] = [0, 0]
    idx = 1
    
    # Rings
    ring_radius = 0.1
    while idx < n:
        # Number of points in this ring
        # Approximate circumference / spacing
        # spacing ~ 2 * radius_est
        # Let's just try increasing numbers of points
        num_points = int(2 * np.pi * ring_radius / 0.25) 
        if num_points < 1: num_points = 1
        
        for k in range(num_points):
            if idx >= n:
                break
            theta = 2 * np.pi * k / num_points + rng.uniform(-0.1, 0.1)
            r = ring_radius * rng.uniform(0.95, 1.05)
            centers[idx] = [r * np.cos(theta), r * np.sin(theta)]
            idx += 1
        
        ring_radius += 0.15
        
    # Add random perturbation
    centers += rng.normal(0, 0.01, centers.shape)
    return centers

def _get_golden_initialization(n, rng):
    """
    Generate an initialization using golden angle spiral.
    """
    centers = np.zeros((n, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(n):
        r = np.sqrt(i / n) * 0.9
        theta = i * golden_angle + rng.uniform(-0.1, 0.1)
        centers[i] = [r * np.cos(theta), r * np.sin(theta)]
        
    centers += rng.normal(0, 0.01, centers.shape)
    return centers

def _get_random_initialization(n, rng):
    """
    Generate random initialization.
    """
    centers = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(rng.random()) * 0.9
        theta = rng.random() * 2 * np.pi
        centers[i] = [r * np.cos(theta), r * np.sin(theta)]
    return centers

def _local_search(centers, time_limit, rng, step_size=0.05):
    """
    Simple local search: random perturbations, accept if score improves.
    """
    start_time = time.time()
    current = centers.copy()
    current_score = _compute_score(current)
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        i = rng.randint(N)
        # Random perturbation
        delta = rng.normal(0, step_size, 2)
        new_point = current[i] + delta
        
        # Check if new point is roughly inside (optimization will fix, but avoid huge jumps)
        if np.linalg.norm(new_point) > 1.5:
            continue
            
        current[i] = new_point
        
        new_score = _compute_score(current)
        
        if new_score > current_score:
            current_score = new_score
            # Maybe reduce step size slightly if we are making progress? 
            # Or keep it constant for exploration.
        else:
            # Revert
            current[i] = current[i] - delta
            # If we haven't improved in a while, reduce step size?
            # For simplicity, keep constant or decay slowly.
            
    return current, current_score

def _polish(centers, time_limit, beta=1000):
    """
    High precision polish using L-BFGS-B with high beta.
    """
    start_time = time.time()
    
    def fun(x):
        val, g = _objective_and_grad(x, N, beta)
        return val, g
        
    try:
        res = minimize(fun, centers.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 100, 'gtol': 1e-10})
        if res.success:
            new_centers = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_centers)):
                return new_centers
    except Exception:
        pass
        
    return centers

def construct():
    """Construct the best packing found within the time budget."""
    deadline = time.time() + 145.0
    rng = np.random.RandomState(42)
    
    best_centers = None
    best_score = 0.0
    
    # List of initializers
    inits = []
    
    # Ring init
    inits.append(_get_ring_initialization(N, rng))
    
    # Golden angle init
    inits.append(_get_golden_initialization(N, rng))
    
    # Random inits
    for _ in range(3):
        inits.append(_get_random_initialization(N, rng))
        
    # Hex-like init (simple grid)
    hex_init = np.zeros((N, 2))
    # ... (Skipping complex hex generation for brevity in thought, using random/ring/golden as primary)
    
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        remaining = deadline - time.time()
        
        # Phase 1: Local Search
        # Allocate 50% of remaining time to local search
        t_ls = remaining * 0.5
        if t_ls < 1.0:
            continue
            
        ls_centers, ls_score = _local_search(init_centers, t_ls, rng, step_size=0.02)
        
        # Phase 2: Polish
        t_polish = remaining - t_ls - 2.0
        if t_polish < 1.0:
            continue
            
        polished_centers = _polish(ls_centers, t_polish, beta=500)
        polished_score = _compute_score(polished_centers)
        
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished_centers
            
            # If we found a very good solution, we can try to polish it further with even higher beta
            if best_score > 0.125:
                if time.time() < deadline - 5:
                    final_polish = _polish(best_centers, deadline - time.time() - 2, beta=2000)
                    final_score = _compute_score(final_polish)
                    if final_score > best_score:
                        best_score = final_score
                        best_centers = final_polish
                break

    if best_centers is None:
        # Fallback
        best_centers = _get_random_initialization(N, rng)
        
    return best_centers
# EVOLVE-BLOCK-END
