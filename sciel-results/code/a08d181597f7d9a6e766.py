"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math

N = 150
# Deadline guard: leave a safety margin (10 s) under the 120 s limit
_DEADLINE = time.time() + 110.0

def _energy(pts):
    """Vectorized sum_{i<j} log(1 / |x_i - x_j|) using math.fsum."""
    # pts is (N, 3)
    # diff is (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    # dist is (N, N)
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal and upper triangle to avoid log(0) and double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    # Filter out any extremely small distances (shouldn't happen, but safe)
    d = d[d > 1e-15]
    if len(d) == 0:
        return 1e18
    terms = -np.log(d)
    return math.fsum(terms)

def _grad(pts):
    """Vectorized gradient of E wrt each point (tangent-projected)."""
    m = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]        # (m,m,3)
    dist = np.linalg.norm(diff, axis=2)              # (m,m)
    # Avoid division by zero
    np.fill_diagonal(dist, 1.0)
    inv_d2 = 1.0 / (dist * dist)                     # (m,m)
    # Zero out diagonal
    np.fill_diagonal(inv_d2, 0.0)
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Using einsum: 'ij,ijk->ik'
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)     # (m,3)
    # Project to tangent: remove radial component
    # grad_tan = grad - <grad, x> x
    radial_comp = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial_comp * pts
    return grad

def _retract(pts):
    """Project points back to the sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _gd_momentum(pts, max_iters=500000, tol=1e-12):
    """Riemannian Gradient Descent with Momentum."""
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        g = _grad(pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Nesterov momentum
        v = momentum * v - lr * g
        new_pts = pts + v
        new_pts = _retract(new_pts)
        
        new_E = _energy(new_pts)
        
        if new_E < E:
            pts = new_pts
            E = new_E
            if E < best_E:
                best_E = E
                best_pts = pts.copy()
            # Decay learning rate slowly
            if it % 1000 == 0:
                lr *= 0.99
        else:
            # If step was bad, reduce learning rate and reset momentum
            lr *= 0.5
            v = 0.0
            # Try a smaller step
            new_pts = pts - 0.5 * lr * g
            new_pts = _retract(new_pts)
            new_E = _energy(new_pts)
            if new_E < E:
                pts = new_pts
                E = new_E
                if E < best_E:
                    best_E = E
                    best_pts = pts.copy()

    return best_pts, best_E

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def construct():
    t0 = time.time()
    
    # 1. Start with Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    best_pts, best_E = _gd_momentum(pts0, max_iters=100000)
    
    # 2. Multiple random restarts to escape local minima
    seeds = [42, 123, 999, 2023, 777]
    for seed in seeds:
        if time.time() > _DEADLINE - 20:
            break
        p = _random_sphere(N, seed)
        p2, E2 = _gd_momentum(p, max_iters=50000)
        if E2 < best_E:
            best_pts, best_E = p2, E2
    
    # 3. Final high-precision polish
    if time.time() < _DEADLINE - 5:
        # Use a very tight tolerance and small initial lr for polish
        # We can just run GD again with a smaller lr
        # Or use L-BFGS if available and fast enough? 
        # Let's stick to GD with a very small lr for the last few seconds
        polished_pts, polished_E = _gd_momentum(best_pts, max_iters=50000)
        if polished_E < best_E:
            best_pts, best_E = polished_pts, polished_E

    # Final normalization
    best_pts = _retract(best_pts)
    
    return best_pts
