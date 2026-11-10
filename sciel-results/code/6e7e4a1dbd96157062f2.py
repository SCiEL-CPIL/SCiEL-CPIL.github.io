"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100
# Time limit with safety margin
TIME_LIMIT = 115.0  # seconds, leaving 5s buffer for the 120s limit


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Mask diagonal and lower triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Filter out any distances that are too small (shouldn't happen but safety)
    d = d[d > 1e-15]
    if len(d) == 0:
        return float('inf')
    return float(np.sum(np.log(1.0 / d)))


def _log_energy_fsum(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Use math.fsum for the sum
    vals = [math.log(1.0 / di) for di in d]
    return math.fsum(vals)


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient(pts):
    """
    Compute gradient of the logarithmic energy on the sphere.
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    
    For a point xi, the gradient contribution from pair (i,j) is:
    d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    
    But we need the gradient on the sphere, so we project to the tangent space:
    grad_sphere = grad_euclidean - (grad_euclidean . xi) * xi
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-20:
                continue
            dist = math.sqrt(dist_sq)
            # Gradient of -log(dist) w.r.t. xi is -diff/dist^2
            g = -diff / dist_sq
            # Project to tangent space at xi
            grad[i] += g - np.dot(g, pts[i]) * pts[i]
            # Gradient w.r.t. xj is +diff/dist^2
            g_j = diff / dist_sq
            grad[j] += g_j - np.dot(g_j, pts[j]) * pts[j]
    
    return grad


def _gradient_vectorized(pts):
    """Vectorized gradient computation."""
    n = len(pts)
    # Pairwise differences: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    # Set diagonal to 1 to avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # For each pair (i,j), the contribution to grad[i] is -(pts[i]-pts[j])/|pts[i]-pts[j]|^2
    # and to grad[j] is +(pts[i]-pts[j])/|pts[i]-pts[j]|^2
    # So grad[i] = sum_{j!=i} (pts[j]-pts[i])/|pts[i]-pts[j]|^2
    #             = sum_{j!=i} -diff[i,j]/dist_sq[i,j]
    
    # inv_dist_sq: 1/dist_sq with 0 on diagonal
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_raw[i] = sum_j inv_dist_sq[i,j] * (pts[j] - pts[i])
    #             = -sum_j inv_dist_sq[i,j] * diff[i,j]
    grad_raw = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space
    # grad_sphere[i] = grad_raw[i] - (grad_raw[i] . pts[i]) * pts[i]
    dot_prods = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad = grad_raw - dot_prods * pts
    
    return grad


def _line_search(pts, grad, energy, max_iter=50):
    """
    Adaptive line search with backtracking.
    Returns new points and new energy.
    """
    step = 0.1
    for _ in range(max_iter):
        new_pts = pts - step * grad
        new_pts = _project_to_sphere(new_pts)
        new_energy = _log_energy(new_pts)
        if new_energy < energy:
            return new_pts, new_energy
        step *= 0.5
    return pts, energy


def _optimize(pts, time_limit, tol=1e-12, max_iter=5000):
    """
    Gradient descent with line search on the sphere.
    """
    start_time = time.time()
    energy = _log_energy(pts)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        grad = _gradient_vectorized(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        pts, energy = _line_search(pts, grad, energy)
    
    return pts, energy


def _random_init(seed=None):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)


def construct():
    start_time = time.time()
    
    # Try multiple initializations: Fibonacci spiral + random starts
    best_pts = None
    best_energy = float('inf')
    
    initializations = [_fibonacci_spiral()]
    
    # Add several random initializations
    for seed in [42, 123, 456, 789, 1000]:
        initializations.append(_random_init(seed))
    
    for init_pts in initializations:
        if time.time() - start_time > TIME_LIMIT - 10.0:
            break
        
        # Optimize from this initialization
        opt_pts, opt_energy = _optimize(init_pts, TIME_LIMIT - (time.time() - start_time))
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # If we have time, do a final polish with tighter tolerance
    if time.time() - start_time < TIME_LIMIT - 10.0:
        polish_pts, polish_energy = _optimize(best_pts, TIME_LIMIT - (time.time() - start_time), tol=1e-14, max_iter=2000)
        if polish_energy < best_energy:
            best_pts = polish_pts
            best_energy = polish_energy
    
    # Ensure points are on the unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral()
    
    return best_pts
