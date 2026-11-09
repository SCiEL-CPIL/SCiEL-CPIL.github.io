"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 150.0
_START_TIME = None


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _compute_radius(centers):
    """Compute the radius for given centers.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    if n < 2:
        boundary = np.min(0.5 - np.abs(centers))
        return float(boundary)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distance
    boundary = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, boundary))


def _fcc_lattice_init(n_points, seed=42):
    """Generate initial points on an FCC lattice within the cube, clipped to n_points."""
    rng = np.random.RandomState(seed)
    
    # FCC lattice points in a unit cube [0,1]^3
    # Generate a grid of FCC points
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # We want points in [-0.5, 0.5]^3
    
    # Let's generate FCC points with spacing
    # For N=100, we want roughly 100 points
    # Try different spacings
    
    best_centers = None
    best_radius = 0.0
    
    for spacing in np.linspace(0.08, 0.2, 15):
        # Generate FCC points
        half = 0.5
        n_side = int(np.ceil(1.0 / spacing)) + 2
        xs = np.arange(-n_side, n_side + 1) * spacing
        xs = xs[(xs >= -half - spacing) & (xs <= half + spacing)]
        
        fcc_points = []
        for i, x in enumerate(xs):
            for j, y in enumerate(xs):
                for k, z in enumerate(xs):
                    # FCC basis offsets
                    offsets = [
                        (0, 0, 0),
                        (0.5 * spacing, 0.5 * spacing, 0),
                        (0.5 * spacing, 0, 0.5 * spacing),
                        (0, 0.5 * spacing, 0.5 * spacing),
                    ]
                    for dx, dy, dz in offsets:
                        px, py, pz = x + dx, y + dy, z + dz
                        if -half <= px <= half and -half <= py <= half and -half <= pz <= half:
                            fcc_points.append((px, py, pz))
        
        if len(fcc_points) < n_points:
            continue
        
        fcc_points = np.array(fcc_points)
        
        # Randomly select n_points from fcc_points
        idx = rng.choice(len(fcc_points), n_points, replace=False)
        centers = fcc_points[idx]
        
        # Compute radius
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers


def _soft_min_radius(centers, temperature=0.01):
    """Soft surrogate for the min radius using log-sum-exp."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    
    # Pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We want to maximize min(dist/2, boundary)
    # Let's use a soft min: -softmax(-values)
    # values = [dist_ij/2 for i<j, boundary_i for each i,k]
    
    # Collect all constraint values
    iu = np.triu_indices(n, 1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    
    boundary = 0.5 - np.abs(centers)
    boundary_vals = boundary.ravel()
    
    all_vals = np.concatenate([pair_dists, boundary_vals])
    
    # Soft min: -logsumexp(-vals/temperature) * temperature
    # Use stable logsumexp
    max_val = np.max(all_vals)
    shifted = -all_vals / temperature
    shifted -= np.max(shifted)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
    soft_min = -log_sum_exp * temperature
    
    return soft_min


def _soft_min_radius_grad(centers, temperature=0.01):
    """Gradient of soft min radius."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(n, 1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    
    boundary = 0.5 - np.abs(centers)
    boundary_vals = boundary.ravel()
    
    all_vals = np.concatenate([pair_dists, boundary_vals])
    
    # Soft min: -logsumexp(-vals/T) * T
    # d/dx soft_min = sum_j (exp(-val_j/T) / sum_k exp(-val_k/T)) * (-dval_j/dx) / 1
    # Actually: soft_min = -T * log(sum_j exp(-val_j/T))
    # d(soft_min)/dx = -T * (1/sum_j exp(-val_j/T)) * sum_j exp(-val_j/T) * (-1/T) * dval_j/dx
    # = sum_j [exp(-val_j/T) / sum_k exp(-val_k/T)] * dval_j/dx
    
    shifted = -all_vals / temperature
    shifted -= np.max(shifted)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    n_pairs = len(pair_dists)
    grad = np.zeros_like(centers)
    
    # Pairwise distances contribution
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        d = pair_dists[idx]
        if d < 1e-12:
            continue
        w = weights[idx]
        # d(pair_dist)/d(centers[i]) = (centers[i] - centers[j]) / (2 * dist) * 2 = (ci-cj)/dist
        # pair_dist = dist/2, so d(pair_dist)/d(ci) = (ci-cj)/(2*dist)
        diff_ij = centers[i] - centers[j]
        dist_ij = 2 * d
        grad[i] += w * diff_ij / (2 * dist_ij)
        grad[j] -= w * diff_ij / (2 * dist_ij)
    
    # Boundary contribution
    boundary_grad = np.zeros_like(centers)
    for idx in range(n * 3):
        i = idx // 3
        k = idx % 3
        w = weights[n_pairs + idx]
        # boundary_val = 0.5 - |c_ik|
        # d(boundary_val)/d(c_ik) = -sign(c_ik)
        c = centers[i, k]
        if c > 1e-12:
            boundary_grad[i, k] -= w
        elif c < -1e-12:
            boundary_grad[i, k] += w
        # else: subgradient, use 0
    
    grad += boundary_grad
    return grad


def _gradient_ascent(centers, max_iter=200, step_size=0.01, temperature=0.01):
    """Gradient ascent on soft min radius."""
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 30:
            break
        
        val = _soft_min_radius(centers, temperature)
        grad = _soft_min_radius_grad(centers, temperature)
        
        centers = centers + step_size * grad
        
        # Project back to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    return best_centers, best_radius


def _local_search(centers, max_iter=500, temperature=0.01):
    """Local search using gradient ascent with adaptive step size."""
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.02
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 30:
            break
        
        grad = _soft_min_radius_grad(centers, temperature)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step size
        new_centers = centers + step_size * grad / grad_norm * 0.1
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        if new_radius > best_radius:
            centers = new_centers
            best_radius = new_radius
            best_centers = centers.copy()
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_centers, best_radius


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Initial configuration from FCC lattice
    centers = _fcc_lattice_init(N, seed=42)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Multiple restarts with different seeds
    for seed in range(42, 42 + 20):
        if _elapsed() > _DEADLINE - 60:
            break
        
        centers = _fcc_lattice_init(N, seed=seed)
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        # Local search
        improved_centers, improved_radius = _local_search(centers, max_iter=300, temperature=0.01)
        if improved_radius > best_radius:
            best_radius = improved_radius
            best_centers = improved_centers.copy()
    
    # Final polish with smaller temperature
    if _elapsed() < _DEADLINE - 45:
        polished_centers, polished_radius = _local_search(best_centers, max_iter=500, temperature=0.005)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers.copy()
    
    # Ensure validity: clip to cube
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
