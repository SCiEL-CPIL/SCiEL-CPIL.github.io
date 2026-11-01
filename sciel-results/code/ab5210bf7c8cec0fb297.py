import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial.distance import pdist

N = 50
DEADLINE = 115.0  # Safety margin for 120s limit

def _compute_score(centers):
    """Compute the packing score for given centers."""
    if len(centers) != N:
        return 0.0
    if np.any(~np.isfinite(centers)):
        return 0.0
    if np.any(np.linalg.norm(centers, axis=1) >= 1.0):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
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
    
    # Sort by distance to center to pick central ones
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = np.sqrt(rng.uniform(0, 0.95**2))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_ring_initialization(n):
    """Generate ring-based initialization."""
    centers = []
    centers.append([0.0, 0.0])
    
    rings = [
        (6, 0.25),
        (12, 0.5),
        (12, 0.75),
        (10, 0.9)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
            
    return np.array(centers[:n])

def _objective_soft_min(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    obj = -(m - (1.0 / beta) * np.log(np.sum(exp_terms)))
    
    # Gradient
    grad = np.zeros_like(centers)
    
    exp_vals = exp_terms
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-8:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
            
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-8:
            continue
        if norms[i] > 1e-8:
            grad[i] -= w * centers[i] / norms[i]
            
    return obj, -grad

def _polish_subgradient(centers, start_time, max_iters=5000, step_size=0.002):
    """
    Direct approach: Sub-gradient ascent on the minimum distance.
    Identifies the tightest constraint and pushes points apart.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        step = step_size
        
        if min_pair_dist / 2.0 < min_bound_dist:
            # Bottleneck is a pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j: i, j = j, i
            d = dists[i, j]
            if d > 1e-8:
                vec = (current[i] - current[j]) / d
                # Move i and j apart
                current[i] += vec * step * 0.5
                current[j] -= vec * step * 0.5
        else:
            # Bottleneck is boundary
            i = np.argmin(bound_dists)
            norm = norms[i]
            if norm > 1e-8:
                vec = current[i] / norm
                # Move i towards center
                current[i] -= vec * step
        
        # Ensure no points go out of bounds (safety check)
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.99
        if np.any(mask):
            current[mask] *= 0.99 / new_norms[mask]
            
        # Check improvement
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
            
    return best_centers, best_score

def _optimize_from_start(centers, start_time):
    """
    Multi-stage optimization:
    1. L-BFGS-B with soft min (beta 50 -> 500)
    2. Sub-gradient polish
    """
    current = centers.copy()
    
    # Stage 1: L-BFGS-B
    betas = [50.0, 200.0, 500.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            res = opt.minimize(
                _objective_soft_min, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            if res.success or np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                # Verify validity
                if np.all(np.isfinite(candidate)):
                    current = candidate
        except Exception:
            pass
            
    # Stage 2: Direct Sub-gradient Polish
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _polish_subgradient(current, start_time, max_iters=5000, step_size=0.002)
        
    return current

def _force_directed_layout(centers, start_time, max_iters=2000):
    """
    Force-directed layout with repulsive potentials.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d > 1e-8:
                    # Repulsive force scaling as 1/d^4
                    f = 1.0 / (d**4)
                    forces[i] += f * diff / d
                    forces[j] -= f * diff / d
        
        # Boundary attraction
        for i in range(N):
            norm = np.linalg.norm(current[i])
            if norm > 1e-8:
                # Pull towards center if close to boundary
                if norm > 0.9:
                    forces[i] -= (norm - 0.9) * current[i] / norm
        
        # Update positions
        step = 0.001 * (1.0 - it / max_iters)
        current += forces * step
        
        # Project back to unit disk
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            current[mask] *= 0.99 / norms[mask]
        
        # Check improvement
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
            
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Try multiple initializations
    inits = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123)
    ]
    
    best_x = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        # Optimize from this initialization
        optimized = _optimize_from_start(init, start_time)
        score = _compute_score(optimized)
        
        if score > best_score:
            best_score = score
            best_x = optimized
        
        # Try force-directed layout as an alternative
        if time.time() < deadline - 20:
            fd_result, fd_score = _force_directed_layout(optimized, start_time, max_iters=1000)
            if fd_score > best_score:
                best_score = fd_score
                best_x = fd_result
    
    # Final validity check and return
    final_points = best_x.reshape(-1, 2)
    if _compute_score(final_points) <= 0:
        # Fallback to hex lattice if optimization failed
        return _get_initial_hex_lattice(N)
        
    return final_points
