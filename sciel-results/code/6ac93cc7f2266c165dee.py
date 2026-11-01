"""Rewritten packing program with improved optimization strategy."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 15.0

def _compute_radius(points):
    """Compute the maximum common radius for given centers."""
    if len(points) < 2:
        return 0.0
    # Distance to boundary
    dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    return min(min_boundary, min_pair)

def _hex_lattice_init():
    """Generate a hexagonal lattice initialization."""
    # Try different spacings to get exactly N points inside the unit circle
    best_points = None
    best_radius = 0.0
    
    for spacing in np.linspace(0.03, 0.12, 20):
        # Generate hexagonal lattice points
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= N:
            # Select N points closest to center
            radius = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radius, kind="stable")[:N]
            points = points[indices]
            
            # Ensure points are inside with margin
            norms = np.linalg.norm(points, axis=1)
            if np.all(norms <= 1.0):
                r = _compute_radius(points)
                if r > best_radius:
                    best_radius = r
                    best_points = points
    
    if best_points is None:
        # Fallback: uniform random
        rng = np.random.default_rng(42)
        points = rng.random((N, 2)) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * 0.95
        best_points = points
    
    return best_points

def _ring_init():
    """Generate a ring-based initialization."""
    # Place circles in concentric rings
    # Center + rings with increasing number of circles
    rings = [1, 6, 12, 18, 24, 30, 36, 42, 48, 53]
    # Adjust to get exactly 300
    while sum(rings) < N:
        rings[-1] += 1
    while sum(rings) > N:
        rings[-1] -= 1
    
    # Radial positions
    n_rings = len(rings)
    radii = np.linspace(0.0, 0.95, n_rings)
    
    points = [np.array([0.0, 0.0])]
    for i in range(1, n_rings):
        n = rings[i]
        r = radii[i]
        angles = np.arange(n) * 2 * np.pi / n
        if i % 2 == 0:
            angles += np.pi / n  # Stagger
        ring_points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        points.append(ring_points)
    
    points = np.vstack(points)
    
    # Trim to exactly N if needed
    if len(points) > N:
        norms = np.linalg.norm(points, axis=1)
        indices = np.argsort(norms, kind="stable")[:N]
        points = points[indices]
    
    return points

def _random_init(seed=42):
    """Generate a random initialization."""
    rng = np.random.default_rng(seed)
    # Rejection sampling to get points inside unit circle
    points = []
    while len(points) < N:
        batch = rng.random((N - len(points), 2)) * 2 - 1
        norms = np.linalg.norm(batch, axis=1)
        valid = batch[norms <= 0.95]
        points.append(valid)
    
    points = np.vstack(points)[:N]
    return points

def _soft_min_loss(points, t=50.0):
    """
    Compute a smooth approximation of the negative minimum distance.
    Uses softmax over distances to create a differentiable surrogate.
    """
    n = len(points)
    points = points.reshape(n, 2)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    # Collect all distances
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min: -log(sum(exp(-t*d))) / t
    # For numerical stability, subtract min
    d_min = np.min(all_dists)
    exp_terms = np.exp(-t * (all_dists - d_min))
    soft_min = d_min - np.log(np.sum(exp_terms)) / t
    
    return -soft_min, exp_terms, iu, diff, dist_matrix, boundary_dists

def _soft_min_grad(points, t=50.0):
    """Compute gradient of the soft min loss."""
    n = len(points)
    points = points.reshape(n, 2)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    d_min = np.min(all_dists)
    exp_terms = np.exp(-t * (all_dists - d_min))
    S = np.sum(exp_terms)
    
    grad = np.zeros_like(points)
    
    # Pairwise gradient
    pair_weights = exp_terms[:len(pair_dists)] / S
    diff_pairs = diff[iu]
    norm_pairs = pair_dists[:, np.newaxis]
    norm_pairs[norm_pairs == 0] = 1e-12
    unit_vecs = diff_pairs / norm_pairs
    
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)
    
    # Boundary gradient
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms = np.linalg.norm(points, axis=1)[:, np.newaxis]
    norms[norms == 0] = 1e-12
    unit_pos = points / norms
    grad += boundary_weights[:, np.newaxis] * unit_pos
    
    return grad

def _local_optimize(points, max_iter=3000, t_start=100.0, t_end=5.0):
    """
    Local optimization using gradient ascent with adaptive temperature.
    """
    n = len(points)
    points = points.copy()
    
    start_time = time.time()
    lr = 0.005
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    for step in range(max_iter):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Anneal temperature
        t = t_start * (t_end / t_start) ** (step / max_iter)
        
        grad = _soft_min_grad(points, t)
        
        # Gradient ascent (we want to maximize min distance)
        points_new = points + lr * grad
        
        # Clamp to unit circle
        norms = np.linalg.norm(points_new, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            points_new[mask] *= 0.999 / norms[mask, np.newaxis]
        
        new_radius = _compute_radius(points_new)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_points = points_new.copy()
            lr *= 1.02
        else:
            lr *= 0.95
        
        points = points_new
        
        if lr < 1e-12:
            break
    
    return best_points

def _simulated_annealing(points, max_iter=5000):
    """
    Simulated annealing to escape local minima.
    """
    n = len(points)
    points = points.copy()
    start_time = time.time()
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    for step in range(max_iter):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Random perturbation
        noise = np.random.randn(n, 2) * T
        points_new = points + noise
        
        # Clamp to unit circle
        norms = np.linalg.norm(points_new, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            points_new[mask] *= 0.999 / norms[mask, np.newaxis]
        
        new_radius = _compute_radius(points_new)
        
        # Accept if better or with probability exp((new - old) / T)
        if new_radius >= best_radius:
            points = points_new
            best_radius = new_radius
            best_points = points_new.copy()
        else:
            delta = new_radius - _compute_radius(points)
            if np.random.random() < np.exp(delta / T):
                points = points_new
        
        T *= cooling
    
    return best_points

def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = []
    
    # Hexagonal lattice
    hex_points = _hex_lattice_init()
    inits.append(hex_points)
    
    # Ring-based
    ring_points = _ring_init()
    inits.append(ring_points)
    
    # Random with different seeds
    for seed in [42, 123, 456]:
        if time.time() > DEADLINE - 30:
            break
        rand_points = _random_init(seed)
        inits.append(rand_points)
    
    # Optimize each initialization
    best_points = None
    best_radius = 0.0
    
    for init_points in inits:
        if time.time() > DEADLINE - 20:
            break
        
        # Local optimization
        opt_points = _local_optimize(init_points, max_iter=2000)
        r = _compute_radius(opt_points)
        
        if r > best_radius:
            best_radius = r
            best_points = opt_points
    
    # Try simulated annealing on the best result
    if best_points is not None and time.time() < DEADLINE - 15:
        sa_points = _simulated_annealing(best_points, max_iter=3000)
        r = _compute_radius(sa_points)
        if r > best_radius:
            best_radius = r
            best_points = sa_points
    
    # Final local polish
    if best_points is not None and time.time() < DEADLINE - 10:
        final_points = _local_optimize(best_points, max_iter=1000)
        r = _compute_radius(final_points)
        if r > best_radius:
            best_radius = r
            best_points = final_points
    
    # Ensure exactly N points
    if best_points is None or len(best_points) != N:
        best_points = hex_points[:N] if len(hex_points) >= N else _random_init(0)
    
    # Final validity check
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        max_norm = np.max(norms)
        if max_norm > 1.0:
            best_points = best_points * (0.999 / max_norm)
    
    return best_points
# EVOLVE-BLOCK-END
