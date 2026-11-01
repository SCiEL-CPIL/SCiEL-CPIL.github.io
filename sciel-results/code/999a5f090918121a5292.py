"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

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
    # If m is very negative, exp(-beta*(v-m)) might overflow if v-m is large positive?
    # v-m >= 0. So exp(-beta * positive) is small.
    # The term with v=m is exp(0)=1.
    # So sum is >= 1. log is safe.
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
    # d(soft_min)/dx = -sum(weights * dx/dx_i) where weights are softmax of -beta*v
    # We want to minimize -soft_min, so grad = sum(weights * dx/dx_i)
    
    v_min = np.min(vals)
    exp_vals = np.exp(-beta * (vals - v_min))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part
    # weight_k corresponds to pair (i_k, j_k)
    # d(pair_dist_k / 2) / d(centers[i]) = (centers[i] - centers[j]) / (2 * dist)
    # d(pair_dist_k / 2) / d(centers[j]) = (centers[j] - centers[i]) / (2 * dist)
    
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            # Derivative of dist/2 w.r.t c_i is vec/2
            # Derivative of dist/2 w.r.t c_j is -vec/2
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
            
    # Boundary part
    # weight_k corresponds to center i_k
    # d(1 - norm) / d(c_i) = -c_i / norm
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        if norms[i] > 1e-8:
            grad[i] -= w * centers[i] / norms[i]
            
    return obj, grad

def _repulsion_objective_and_grad(params, n):
    """
    Minimize repulsion energy to spread points.
    E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2
    """
    centers = params.reshape(n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # Avoid division by zero
    pair_dists_safe = np.maximum(pair_dists, 1e-6)
    bound_dists_safe = np.maximum(bound_dists, 1e-6)
    
    # Energy
    obj = np.sum(1.0 / (pair_dists_safe ** 2)) + np.sum(1.0 / (bound_dists_safe ** 2))
    
    grad = np.zeros_like(centers)
    
    # Gradient of 1/d^2 is -2/d^3 * d(d)/dx
    # d(d)/dx_i = (x_i - x_j)/d
    # So d(1/d^2)/dx_i = -2/d^3 * (x_i - x_j)/d = -2*(x_i - x_j)/d^4
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        d = pair_dists[k]
        if d < 1e-6:
            continue
        vec = centers[i] - centers[j]
        grad[i] += -2.0 * vec / (d ** 4)
        grad[j] += 2.0 * vec / (d ** 4)
        
    # Gradient of 1/(1-r)^2 is 2/(1-r)^3 * d(1-r)/dx = 2/(1-r)^3 * (-x/r)
    for i in range(n):
        r = norms[i]
        b = bound_dists[i]
        if b < 1e-6:
            continue
        if r > 1e-8:
            grad[i] += 2.0 * (-centers[i] / r) / (b ** 3)
            
    return obj, grad

def _hexagonal_init(n, radius=0.12, perturb=0.01, rng=None):
    """Generate hexagonal lattice initialization."""
    if rng is None:
        rng = np.random.RandomState(42)
    spacing = 2 * radius
    sqrt3 = np.sqrt(3)
    points = []
    max_k = int(np.ceil(1.0 / spacing)) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - radius:
                points.append((x, y))
    
    if len(points) < n:
        # Fill with random if lattice is too small
        while len(points) < n:
            r = np.sqrt(rng.random()) * (1.0 - radius)
            theta = rng.random() * 2 * np.pi
            points.append((r * np.cos(theta), r * np.sin(theta)))
            
    points = np.array(points[:n])
    # Add perturbation to break symmetry
    points += rng.normal(0, perturb, points.shape)
    return points

def _random_init(n, radius=0.12, rng=None):
    """Generate random initialization."""
    if rng is None:
        rng = np.random.RandomState()
    centers = []
    for _ in range(n):
        r = np.sqrt(rng.random()) * (1.0 - radius)
        theta = rng.random() * 2 * np.pi
        centers.append((r * np.cos(theta), r * np.sin(theta)))
    return np.array(centers)

def _optimize_phase(centers, time_limit, mode='repulsion'):
    """
    Run L-BFGS-B optimization.
    mode: 'repulsion' or 'maxmin'
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    if mode == 'repulsion':
        def fun(x):
            val, g = _repulsion_objective_and_grad(x, N)
            return val, g
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'gtol': 1e-6})
    else:
        # For maxmin, we ramp up beta
        # This function is called multiple times with increasing beta
        pass 
        
    return res.x.reshape(N, 2)

def _optimize_maxmin(centers, time_limit, beta_start=10.0, beta_end=500.0):
    """
    Progressive beta optimization for max-min.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Geometric progression of beta
    betas = np.geomspace(beta_start, beta_end, num=10)
    
    for beta in betas:
        if time.time() - start_time > time_limit - 2:
            break
            
        def fun(x):
            val, g = _objective_and_grad(x, N, beta)
            return val, g
            
        try:
            res = minimize(fun, current.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': 50, 'gtol': 1e-8})
            if res.success or res.fun < _compute_score(current):
                # Check validity
                new_centers = res.x.reshape(N, 2)
                if np.all(np.isfinite(new_centers)):
                    # Ensure points are roughly inside (L-BFGS-B doesn't have bounds by default here, 
                    # but the penalty/soft-min handles it. 
                    # However, to be safe, we can project or just check score)
                    current = new_centers
        except Exception:
            break
            
    return current

def construct():
    """Construct the best packing found within the time budget."""
    deadline = time.time() + 145.0
    rng = np.random.RandomState(42)
    
    best_centers = None
    best_score = 0.0
    
    # 1. Generate Initializations
    inits = []
    
    # Hexagonal lattices with different rotations and perturbations
    for angle in [0, 0.1, 0.2, 0.5, 1.0, 1.5]:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        base = _hexagonal_init(N, radius=0.12, perturb=0.02, rng=rng)
        rotated = base @ rot.T
        inits.append(rotated)
        
    # Random inits
    for _ in range(5):
        inits.append(_random_init(N, radius=0.12, rng=rng))
        
    # 2. Optimization Loop
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        remaining = deadline - time.time()
        
        # Phase 1: Repulsion (Quick spread)
        # Allocate 10% of remaining time
        t_rep = min(5.0, remaining * 0.1)
        if t_rep > 1.0:
            try:
                spread_centers = _optimize_phase(init_centers, t_rep, mode='repulsion')
            except:
                spread_centers = init_centers
        else:
            spread_centers = init_centers
            
        # Phase 2: Max-Min Polish
        # Allocate remaining time minus safety margin
        t_mm = remaining - t_rep - 5.0
        if t_mm < 2.0:
            continue
            
        try:
            polished = _optimize_maxmin(spread_centers, t_mm, beta_start=20.0, beta_end=1000.0)
        except Exception:
            polished = spread_centers
            
        score = _compute_score(polished)
        
        if score > best_score:
            best_score = score
            best_centers = polished
            
        # If we have a very good score, we can stop early or refine
        if best_score > 0.1258:
            # Try one more high-precision polish on the best found so far
            if time.time() < deadline - 10:
                try:
                    final_polish = _optimize_maxmin(best_centers, deadline - time.time() - 5, beta_start=1000.0, beta_end=5000.0)
                    final_score = _compute_score(final_polish)
                    if final_score > best_score:
                        best_score = final_score
                        best_centers = final_polish
                except:
                    pass
            break

    if best_centers is None:
        best_centers = _hexagonal_init(N, radius=0.12, perturb=0.01, rng=rng)
        
    return best_centers
# EVOLVE-BLOCK-END
