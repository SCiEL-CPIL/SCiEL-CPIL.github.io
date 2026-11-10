"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Deadline guard: return best result before 150s wall-clock
_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    n = len(points)
    # Pairwise distances via broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))


def _energy_and_grad(points):
    """Compute energy and gradient on the sphere.
    
    Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient for point i: sum_{j!=i} (xi - xj) / |xi-xj|^2  (then project to tangent)
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n,n,3)
    dists = np.linalg.norm(diff, axis=2)  # (n,n)
    # Avoid division by zero
    dists = np.maximum(dists, 1e-15)
    
    # grad_i = -sum_{j!=i} (xi-xj) / |xi-xj|^2
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    inv_d2[~mask] = 0.0
    
    # grad[i] = -sum_j diff[i,j] * inv_d2[i,j]
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Project gradient to tangent space (remove radial component)
    grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = float(np.sum(np.log(1.0 / d)))
    
    return energy, grad_tangent


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, 1e-15)


def _adam_tangent(points, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iters=5000):
    """Adam optimizer on the tangent space of the sphere."""
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    # Adam state
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    t = 0
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    for it in range(max_iters):
        if _remaining_time() < 10.0:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update in tangent space
        update = -lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project update to tangent space
        update_tangent = update - np.sum(update * points, axis=1, keepdims=True) * points
        
        # Move and project back to sphere
        points = points + update_tangent
        points = _project_to_sphere(points)
        
        # Adaptive learning rate: reduce if energy increases significantly
        if energy > best_energy + 10.0:
            lr *= 0.5
    
    return best_points, best_energy


def _lbfgs_tangent(points, max_iters=2000, tol=1e-10):
    """L-BFGS optimization on the tangent space using scipy."""
    from scipy.optimize import minimize
    
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol})
    
    if result.success or result.fun < best_energy:
        best_points = _project_to_sphere(result.x.reshape(n, 3))
        best_energy = float(result.fun)
    
    return best_points, best_energy


def _high_precision_polish(points, max_iters=5000, tol=1e-14):
    """High-precision final polish with very tight tolerances."""
    from scipy.optimize import minimize
    
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol,
                               'maxls': 50, 'maxcor': 20})
    
    if result.success or result.fun < best_energy:
        best_points = _project_to_sphere(result.x.reshape(n, 3))
        best_energy = float(result.fun)
    
    return best_points, best_energy


def _spectral_init(n, rng, n_random=2000):
    """Generate initial points using spectral method on sphere.
    
    Uses the fact that the first few eigenvectors of the Laplacian
    on the sphere give approximately uniform distributions.
    """
    # Generate random points on sphere
    random_pts = _random_sphere(n_random, rng)
    
    # Compute pairwise distances
    diff = random_pts[:, np.newaxis, :] - random_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Build a k-NN graph
    k = 20
    # For each point, find k nearest neighbors
    knn_indices = np.argsort(dists, axis=1)[:, 1:k+1]
    
    # Build adjacency matrix
    adj = np.zeros((n_random, n_random))
    for i in range(n_random):
        for j in knn_indices[i]:
            adj[i, j] = 1.0
    
    # Make symmetric
    adj = np.maximum(adj, adj.T)
    
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Get eigenvectors (skip trivial first one)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Take eigenvectors 1 through n (0-indexed, skip 0)
        # These give a low-dimensional embedding
        target_dim = min(n, n_random - 1)
        selected = eigvecs[:, 1:1+target_dim]
    except:
        # Fallback to random
        return _random_sphere(n, rng)
    
    # Normalize rows to get points on sphere
    norms = np.linalg.norm(selected, axis=1, keepdims=True)
    selected = selected / np.maximum(norms, 1e-15)
    
    # Take first n points or sample n from them
    if len(selected) >= n:
        indices = rng.choice(len(selected), size=n, replace=False)
        pts = selected[indices]
    else:
        pts = selected
    
    return _project_to_sphere(pts)


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral as initial guess
    best_points = _fibonacci_sphere(N)
    best_energy = _energy(best_points)
    
    # Also try spectral initialization
    try:
        spectral_pts = _spectral_init(N, rng)
        spectral_energy = _energy(spectral_pts)
        if spectral_energy < best_energy:
            best_energy = spectral_energy
            best_points = spectral_pts.copy()
    except:
        pass
    
    # Multiple restarts with Adam on tangent space
    n_restarts = 8
    
    # Perturbation scales for diversity
    perturbation_scales = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    
    for restart in range(n_restarts):
        if _remaining_time() < 20.0:
            break
        
        if restart == 0:
            # First restart: use current best
            init_points = best_points.copy()
        else:
            # Subsequent restarts: random perturbations of the best found
            scale = perturbation_scales[restart]
            noise = rng.standard_normal((N, 3)) * scale
            init_points = _project_to_sphere(best_points + noise)
        
        # Adam optimization with slightly varied learning rates
        lr = 0.01 * (0.8 ** restart)
        pts, energy = _adam_tangent(
            init_points, 
            lr=lr, 
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-8,
            max_iters=3000
        )
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # First L-BFGS polish if time permits
    if _remaining_time() > 30.0:
        pts, energy = _lbfgs_tangent(best_points, max_iters=2000, tol=1e-12)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # High-precision final polish
    if _remaining_time() > 15.0:
        pts, energy = _high_precision_polish(best_points, max_iters=5000, tol=1e-15)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Verify no points are too close
    dists = np.linalg.norm(best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = dists[iu].min()
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_sphere(N)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
