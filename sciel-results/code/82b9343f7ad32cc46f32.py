"""Minimize logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(pts):
    """Fast energy computation for optimization (no fsum, vectorized)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask out diagonal and upper triangle
    mask = ~np.eye(n, dtype=bool)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points, projected onto sphere."""
    n = len(pts)
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid division by zero
    dists_safe = np.maximum(dists, 1e-15)
    
    # Gradient of log(1/d) = -1/d * d/dx_i (1/d) ... let's derive properly
    # Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i = -sum_{j != i} (1/|x_i - x_j|) * (x_i - x_j)/|x_i - x_j|^2
    #          = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^3
    
    # Compute sum over j for each i
    # inv_dist3[i, j] = 1 / |x_i - x_j|^3
    inv_dist3 = np.zeros_like(dists)
    mask = ~np.eye(n, dtype=bool)
    inv_dist3[mask] = 1.0 / dists_safe[mask]**3
    
    # grad[i, k] = -sum_j inv_dist3[i,j] * diff[i,j,k]
    grad = -np.sum(inv_dist3[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient onto tangent space of sphere
    # proj(v) = v - (v . x_i) * x_i
    dot = np.sum(grad * pts, axis=1)
    grad = grad - dot[:, np.newaxis] * pts
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple initial configurations
    def fibonacci_spiral(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_uniform(n, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        pts = rng.standard_normal((n, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    
    def sobol_like(n, seed=0):
        # Use quasi-random points via Halton sequence approximation
        from scipy.spatial import Qhull
        # Simpler: use stratified random
        rng = np.random.default_rng(seed)
        # Generate points on sphere using inverse CDF for uniform distribution
        z = rng.uniform(-1, 1, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(1 - z**2)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    
    # Try multiple starting points
    best_energy = float('inf')
    best_pts = None
    
    initial_configs = [
        fibonacci_spiral(N),
        random_uniform(N, seed=42),
        random_uniform(N, seed=123),
        random_uniform(N, seed=777),
        sobol_like(N, seed=0),
        sobol_like(N, seed=1),
    ]
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 15.0:
            break
        
        # Optimize using L-BFGS-B with constraints
        # We parameterize on the sphere directly and project after each step
        # Use scipy minimize with a custom approach: optimize in R^3 and project
        
        def objective_and_grad(x):
            # x is flattened (N*3,)
            pts = x.reshape(N, 3)
            # Normalize to sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts = pts / norms
            
            e = _log_energy_fast(pts)
            g = _log_energy_grad(pts)
            # Gradient with respect to x (before normalization)
            # Chain rule: dE/dx = (I - x x^T / |x|^2) * dE/dx_projected ... 
            # Actually, since we normalize, the gradient in x-space is:
            # g_x = (I - x x^T / |x|^2) * g_projected / |x|
            # But for simplicity, let's just use the projected gradient directly
            # and let the optimizer handle it. Since we're on a manifold, we'll
            # use a different approach.
            
            return e, g.flatten()
        
        # Alternative: use SLSQP or trust-constr with sphere constraint
        # Let's use a simpler approach: iterative gradient descent with projection
        
        pts = init_pts.copy()
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            g = _log_energy_grad(pts)
            # Gradient descent with adaptive step size
            pts_new = pts - lr * g
            # Project back to sphere
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_new = pts_new / norms
            
            # Check for improvement
            e_new = _log_energy_fast(pts_new)
            e_old = _log_energy_fast(pts)
            
            if e_new < e_old:
                pts = pts_new
                lr = min(lr * 1.05, 0.5)
            else:
                lr = max(lr * 0.5, 1e-6)
                if lr < 1e-6:
                    break
        
        # Final refinement with scipy
        if time.time() < deadline - 10.0:
            x0 = pts.flatten()
            
            def obj(x):
                p = x.reshape(N, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                p = p / norms
                return _log_energy_fast(p)
            
            def grad(x):
                p = x.reshape(N, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                p = p / norms
                g = _log_energy_grad(p)
                return g.flatten()
            
            try:
                res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                              options={'maxiter': 500, 'gtol': 1e-8})
                if res.success or res.fun < best_energy:
                    p = res.x.reshape(N, 3)
                    norms = np.linalg.norm(p, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    p = p / norms
                    e = _log_energy_fast(p)
                    if e < best_energy:
                        best_energy = e
                        best_pts = p.copy()
            except Exception:
                pass
        
        # Update best
        e = _log_energy_fast(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Final verification with exact energy
    if best_pts is not None:
        # Ensure no points are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-9:
                    # Perturb
                    best_pts[j] += np.random.randn(3) * 1e-5
                    best_pts[j] /= np.linalg.norm(best_pts[j])
        
        # Normalize
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = fibonacci_spiral(N)
    
    return best_pts
