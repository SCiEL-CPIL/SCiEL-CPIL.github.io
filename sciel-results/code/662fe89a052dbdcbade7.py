"""Packing 50 equal circles in a unit circle using soft-min surrogate + gradient optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the maximum radius r for which circles centered at 'centers' 
    are disjoint and inside the unit circle.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    """
    if centers.size == 0:
        return 0.0
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints)
    Constraints: d_ij / 2 for pairs, 1 - |c_i| for boundary.
    We want to maximize the minimum of these, so we minimize the negative soft-min.
    
    Soft min definition:
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    This is a smooth approximation of min(x_i).
    
    To improve numerical stability, we shift x by the current minimum:
    Let m = min(x).
    soft_min(x) = m - 1/beta * log(sum(exp(-beta * (x_i - m))))
    """
    centers = params.reshape(n, 2)
    
    # 1. Compute pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Compute boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # 3. Combine all constraints into a single vector
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # 4. Compute Soft Min
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    obj_val = -(m - (1.0 / beta) * np.log(S))
    
    # 5. Compute Gradient
    weights = exp_terms / S
    
    grad_centers = np.zeros_like(centers)
    
    # Pair contributions
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            grad_centers[i] -= w * 0.5 * vec
            grad_centers[j] += w * 0.5 * vec

    # Boundary contributions
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm > 1e-8:
            vec = centers[i] / norm
            grad_centers[i] += w * vec
            
    return obj_val, grad_centers.flatten()

def _get_ring_initialization(n):
    """
    Generate a ring-based initialization.
    Approximate optimal distribution for N=50:
    Center: 1
    Ring 1: 6
    Ring 2: 12
    Ring 3: 12
    Ring 4: 9
    Ring 5: 10
    Total: 50
    """
    centers = []
    centers.append([0.0, 0.0])
    
    rings = [
        (6, 0.22),
        (12, 0.44),
        (12, 0.66),
        (9, 0.82),
        (10, 0.92)
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

def _get_hex_lattice_initialization(n):
    """
    Generate a hexagonal lattice initialization.
    """
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
                
    if len(points) < n:
        return _get_ring_initialization(n)
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(10000):
        if len(centers) >= n:
            break
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y > 0.95**2:
            continue
        if centers:
            pts = np.array(centers)
            dists = np.hypot(pts[:,0] - x, pts[:,1] - y)
            if np.min(dists) < 0.15:
                continue
        centers.append([x, y])
    
    while len(centers) < n:
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-0.9, 0.9)
        centers.append([x, y])
        
    return np.array(centers[:n])

def _force_directed_relax(centers, time_budget=5.0):
    """
    Pre-processing step: use force-directed relaxation to resolve overlaps
    and push points to a valid configuration before L-BFGS.
    """
    centers = centers.copy()
    start_time = time.time()
    step = 0.01
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    while time.time() - start_time < time_budget:
        n = len(centers)
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                d = centers[i] - centers[j]
                dist = np.hypot(d[0], d[1])
                if dist < 1e-10:
                    d = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
                    dist = np.hypot(d[0], d[1])
                force_mag = 1.0 / (dist * dist + 1e-6)
                force = force_mag * d / dist
                forces[i] += force
                forces[j] -= force
        
        # Boundary constraint
        for i in range(n):
            r = np.hypot(centers[i,0], centers[i,1])
            if r > 0.95:
                forces[i] -= 0.5 * centers[i] / (r + 1e-10)
        
        new_centers = centers + step * forces
        new_radii = np.hypot(new_centers[:,0], new_centers[:,1])
        mask = new_radii > 1.0
        if np.any(mask):
            new_centers[mask] = new_centers[mask] / new_radii[mask][:, None] * 0.999
        
        new_score = _compute_score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers
            step *= 1.05
        else:
            step *= 0.9
            centers = best_centers.copy()
            
        if step < 1e-8:
            break
            
    return best_centers, best_score

def _optimize_from_start(centers, time_budget=20.0):
    """
    Optimize starting from a given configuration using L-BFGS with increasing beta.
    """
    start_time = time.time()
    current_centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    # Force-directed relaxation first
    current_centers, _ = _force_directed_relax(current_centers, time_budget=min(5.0, time_budget * 0.2))
    score = _compute_score(current_centers)
    if score > best_score:
        best_score = score
        best_centers = current_centers.copy()
        
    # L-BFGS optimization with increasing beta
    betas = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0, 100000.0]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 1.0:
            break
            
        def fun_and_grad(x):
            val, grad = _soft_min_obj_and_grad(x, N, beta)
            return val, grad
            
        try:
            result = minimize(
                fun_and_grad,
                current_centers.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            current_centers = result.x.reshape(N, 2)
        except Exception:
            break
            
        score = _compute_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
    return best_centers, best_score

def construct():
    """Main function to construct the packing."""
    start_time = time.time()
    
    # Generate multiple diverse initializations
    initializations = []
    
    # Ring structure
    initializations.append(_get_ring_initialization(N))
    
    # Hex lattice
    initializations.append(_get_hex_lattice_initialization(N))
    
    # Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_get_random_initialization(N, seed=seed))
    
    best_centers = None
    best_score = 0.0
    
    for i, init in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10.0:
            break
            
        remaining = DEADLINE - elapsed - 5.0
        time_per_opt = max(1.0, min(remaining / (len(initializations) - i), 20.0))
        
        optimized, score = _optimize_from_start(init, time_budget=time_per_opt)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
    
    if best_centers is None:
        best_centers = initializations[0]
    
    # Ensure validity
    radii = np.hypot(best_centers[:,0], best_centers[:,1])
    mask = radii > 1.0
    if np.any(mask):
        best_centers[mask] = best_centers[mask] / radii[mask][:, None] * 0.999
        
    return best_centers
# EVOLVE-BLOCK-END
