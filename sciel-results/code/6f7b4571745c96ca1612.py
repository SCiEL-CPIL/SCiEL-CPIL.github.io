"""Rewritten program: minimize logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize, L_BFGS_B
from scipy.spatial.distance import cdist

N = 150
TIME_LIMIT = 115.0  # safety margin below 120s

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)."""
    # pts: (N, 3) array on unit sphere
    # Compute pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, 1.0)  # avoid log(0)
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    # Ensure no zero distances
    dists = np.maximum(dists, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in dists)

def _log_energy_and_grad(pts):
    """Compute energy and gradient for a configuration on the sphere.
    
    We parameterize points on the sphere and compute the gradient of the
    energy with respect to the 3D coordinates, then project onto the tangent space.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)  # (n, n)
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-15)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = math.fsum(math.log(1.0 / di) for di in dists[iu])
    
    # Gradient: d/dx_i sum_{j != i} log(1/|xi - xj|)
    # = sum_{j != i} -1/|xi-xj| * (xi - xj)/|xi-xj|
    # = sum_{j != i} -(xi - xj) / |xi-xj|^2
    # But we need to be careful: log(1/d) = -log(d), d = |xi-xj|
    # d(-log d)/dx_i = -1/d * d(d)/dx_i = -1/d * (xi-xj)/d = -(xi-xj)/d^2
    # So grad_i = sum_{j != i} -(xi - xj) / |xi-xj|^2
    
    # Compute gradient efficiently
    # For each pair (i,j), the contribution to grad_i is -(xi-xj)/d^2 and to grad_j is +(xi-xj)/d^2
    # i.e., (xj-xi)/d^2
    
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (xj - xi) / |xi-xj|^2 = sum_j (xj - xi) * inv_d2[i,j]
    # = (sum_j xj * inv_d2[i,j]) - xi * (sum_j inv_d2[i,j])
    sum_xj_inv_d2 = inv_d2 @ pts  # (n, 3)
    sum_inv_d2 = inv_d2.sum(axis=1)  # (n,)
    grad = sum_xj_inv_d2 - pts * sum_inv_d2[:, np.newaxis]
    
    # Project gradient onto tangent space of sphere at each point
    # Tangent projection: g_tangent = g - (g . x) * x
    grad_dot_x = np.sum(grad * pts, axis=1)  # (n,)
    grad = grad - grad_dot_x[:, np.newaxis] * pts
    
    return energy, grad

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def construct():
    start_time = time.time()
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N).astype(np.float64)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Try a few random starts to see if we can find a better initial configuration
    for seed in [1, 2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        rand_pts = _random_sphere(N, seed=seed)
        e = _log_energy(rand_pts)
        if e < best_energy:
            best_energy = e
            best_pts = rand_pts.copy()
    
    # Use L-BFGS-B with a custom approach: we'll optimize on R^3 and normalize
    # Actually, let's use a different approach: optimize with gradient descent
    # on the sphere using the projected gradient
    
    def objective_and_grad(x):
        pts = _normalize(x.reshape(N, 3))
        energy, grad = _log_energy_and_grad(pts)
        # The gradient with respect to x (in R^3) before normalization is more complex
        # Let's just use the energy and let scipy handle it, but we need to keep on sphere
        # Alternative: use the fact that we can move along tangent directions
        return energy, grad.reshape(-1)
    
    # Actually, let's use a simpler approach: iterative optimization
    # using the projected gradient on the sphere
    
    pts = best_pts.copy()
    lr = 0.01  # learning rate
    momentum = 0.9
    vel = np.zeros_like(pts)
    
    max_iter = 5000
    patience = 200
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        energy, grad = _log_energy_and_grad(pts)
        
        # Momentum
        vel = momentum * vel - lr * grad
        pts_new = pts + vel
        
        # Project back to sphere
        pts_new = _normalize(pts_new)
        
        energy_new = _log_energy(pts_new)
        
        if energy_new < energy:
            pts = pts_new
            no_improve = 0
            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts.copy()
        else:
            no_improve += 1
            # Reduce learning rate
            lr *= 0.95
            vel *= 0.5
            pts = pts_new  # still move, but smaller step
        
        if no_improve > patience:
            # Try restarting with a smaller learning rate
            lr = 0.001
            vel = np.zeros_like(pts)
            no_improve = 0
            if time.time() - start_time > TIME_LIMIT - 20:
                break
    
    # Final check
    final_energy = _log_energy(best_pts)
    if final_energy < best_energy:
        best_energy = final_energy
    
    # Ensure validity
    best_pts = _normalize(best_pts)
    
    # Check for points too close
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, 1.0)
    if d.min() < 1e-9:
        # Fall back to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
