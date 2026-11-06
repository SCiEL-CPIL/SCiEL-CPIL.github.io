import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius for the given points.
    r = min(min_pairwise_dist / 2, min_dist_to_boundary)
    """
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def soft_min_surrogate(points, beta=50.0):
    """
    Smooth surrogate for min(min_pair_dist/2, min_boundary_dist).
    Uses log-sum-exp to approximate the minimum smoothly.
    """
    norms = np.linalg.norm(points, axis=1)
    boundary_terms = 1.0 - norms  # N values
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    pair_terms = np.sqrt(dists_sq) / 2.0  # N x N matrix
    
    # Collect all terms into one array
    all_terms = np.concatenate([boundary_terms, pair_terms[np.triu_indices(N, k=1)]])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta approximates min(x) as beta->inf
    # For numerical stability, subtract max first
    m = np.max(all_terms)
    log_sum = np.log(np.sum(np.exp(-beta * (all_terms - m))))
    soft_min_val = m - log_sum / beta
    
    return soft_min_val

def surrogate_grad(points, beta=50.0):
    """
    Gradient of the soft-min surrogate.
    """
    norms = np.linalg.norm(points, axis=1)
    boundary_terms = 1.0 - norms
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_terms = dists / 2.0
    
    all_terms = np.concatenate([boundary_terms, pair_terms[np.triu_indices(N, k=1)]])
    
    m = np.max(all_terms)
    exp_terms = np.exp(-beta * (all_terms - m))
    sum_exp = np.sum(exp_terms)
    
    # Weights: w_k = exp(-beta * x_k) / sum(exp(-beta * x_j))
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(points)
    
    # Boundary terms: d/dx_i (1 - ||x_i||) = -x_i / ||x_i||
    n_boundary = N
    for i in range(N):
        r = norms[i]
        if r > 1e-10:
            grad[i] += weights[i] * (-points[i] / r)
    
    # Pair terms: d/dx_i (||x_i - x_j|| / 2) = (x_i - x_j) / (2 * ||x_i - x_j||)
    iu = np.triu_indices(N, k=1)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[n_boundary + idx]
        d = diff[i, j]
        dist = dists[i, j]
        if dist > 1e-10:
            grad[i] += w * (d / (2.0 * dist))
            grad[j] += w * (-d / (2.0 * dist))
    
    return -grad  # Negate because we want to maximize soft_min, but minimize -soft_min

def _seed_hexagonal():
    """Generate hexagonal lattice seed."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    for _ in range(20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by radius and take innermost N for a compact seed
            order = np.argsort(radius)
            return points[order[:N]]
        spacing *= 0.95
    return None

def _seed_rings():
    """Generate ring-based seed."""
    # Try different ring configurations
    configs = [
        [(0, 0), (0.3, 6), (0.6, 12), (0.85, 18), (1.0, 13)],
        [(0, 0), (0.25, 6), (0.5, 12), (0.75, 18), (0.95, 13)],
        [(0, 0), (0.35, 6), (0.7, 12), (0.95, 18), (1.0, 13)],
    ]
    best_seed = None
    best_score = -1.0
    
    for config in configs:
        pts = []
        for r, count in config:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        if len(pts) >= N:
            seed = np.array(pts[:N])
            score = compute_radius(seed)
            if score > best_score:
                best_score = score
                best_seed = seed
    
    return best_seed

def _seed_random(seed_val):
    """Generate random seed."""
    np.random.seed(seed_val)
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _optimize_from_seed(seed_points, deadline):
    """
    Optimize starting from seed_points using gradient-based method on smooth surrogate,
    then polish with SA on exact score.
    """
    start_time = time.time()
    
    # Phase 1: Gradient-based optimization on smooth surrogate
    x0 = seed_points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Check validity
        if not np.all(np.isfinite(pts)):
            return 1e10
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.01):
            # Heavy penalty for going outside
            excess = np.maximum(0, norms - 1.0)
            return 1e10 * np.sum(excess**2)
        return -soft_min_surrogate(pts, beta=50.0)
    
    def gradient(x):
        pts = x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.01):
            # Gradient of penalty
            grad = np.zeros_like(x)
            for i in range(N):
                if norms[i] > 1.0:
                    grad[2*i:2*i+2] = 2e10 * (norms[i] - 1.0) * pts[i] / norms[i]
            return grad
        return -surrogate_grad(pts, beta=50.0).flatten()
    
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.success or result.fun < objective(x0):
            opt_points = result.x.reshape(N, 2)
            # Project back inside if needed
            norms = np.linalg.norm(opt_points, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                opt_points[mask] = opt_points[mask] / norms[mask, np.newaxis] * 0.999
        else:
            opt_points = seed_points.copy()
    except Exception:
        opt_points = seed_points.copy()
    
    # Phase 2: Simulated Annealing polish on exact score
    current_points = opt_points.copy()
    current_score = compute_radius(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.02
    cooling = 0.995
    step_size = 0.03
    
    while time.time() < deadline - 5.0 and T > 1e-7:
        i = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        new_point = current_points[i] + delta
        
        # Keep inside boundary
        norm = np.linalg.norm(new_point)
        if norm > 1.0:
            new_point = new_point / norm * 0.999
        
        old_val = current_points[i].copy()
        current_points[i] = new_point
        new_score = compute_radius(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta_E = current_score - new_score
            if np.random.rand() < np.exp(-delta_E / T):
                current_score = new_score
            else:
                current_points[i] = old_val
        
        T *= cooling
        step_size *= 0.9995
    
    # Phase 3: Final polish - very small steps to fine-tune
    step_size = 0.005
    T = 0.005
    remaining = deadline - 5.0 - time.time()
    if remaining > 2.0:
        while time.time() < deadline - 2.0 and T > 1e-8:
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, 2)
            new_point = current_points[i] + delta
            norm = np.linalg.norm(new_point)
            if norm > 1.0:
                new_point = new_point / norm * 0.9999
            
            old_val = current_points[i].copy()
            current_points[i] = new_point
            new_score = compute_radius(current_points)
            
            if new_score >= current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                delta_E = current_score - new_score
                if np.random.rand() < np.exp(-delta_E / T):
                    current_score = new_score
                else:
                    current_points[i] = old_val
            
            T *= 0.998
            step_size *= 0.9998
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    seeds = []
    
    # Hexagonal seed
    hex_seed = _seed_hexagonal()
    if hex_seed is not None:
        seeds.append(hex_seed)
    
    # Ring seeds
    ring_seed = _seed_rings()
    if ring_seed is not None:
        seeds.append(ring_seed)
    
    # Random seeds
    for s in [42, 123, 456]:
        seeds.append(_seed_random(s))
    
    best_points = None
    best_score = -1.0
    
    for seed in seeds:
        if time.time() > deadline - 15.0:
            break
        
        result = _optimize_from_seed(seed, deadline)
        score = compute_radius(result)
        
        if score > best_score:
            best_score = score
            best_points = result.copy()
    
    if best_points is None:
        # Fallback: return hexagonal or random
        if seeds:
            best_points = seeds[0]
        else:
            best_points = _seed_random(42)
    
    return best_points
