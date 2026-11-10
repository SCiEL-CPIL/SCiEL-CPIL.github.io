"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize

DEADLINE = 120.0
SAFETY_MARGIN = 0.88  # use 88% of time

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _log_energy_vec(pts):
    """Vectorized log energy for faster evaluation."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask out diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    if np.any(d < 1e-15):
        return 1e30
    return -np.sum(np.log(d))

def _log_energy_grad(pts):
    """Compute energy and gradient. E = -sum log(d_ij). grad_i = -sum_{j!=i} (x_i - x_j)/d_ij^2"""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-30:
                return 1e30, grad
            d = np.sqrt(d2)
            energy -= log(d)
            force = diff / d2  # (x_i - x_j) / d^2
            grad[i] -= force  # dE/dx_i = -force
            grad[j] += force  # dE/dx_j = +force
    return energy, grad

def _fibonacci_spiral(n, offset=0.0):
    """Golden-angle spiral points on unit sphere with optional phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Random points on unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _repulsion_refine(pts, n_iters=2000, lr=0.01, deadline=None):
    """Iterative repulsion with sphere projection."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Compute forces: F_i = sum_{j!=i} (x_i - x_j) / d_ij^2
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-12:
                    forces[i] += 1000.0 * diff  # strong repulsion if too close
                else:
                    forces[i] += diff / d2
        pts += lr * forces
        # Project back to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE * SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    inits = []
    # Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    # Fibonacci with different offsets
    for k in range(1, 4):
        inits.append(_fibonacci_spiral(N, offset=k * 0.5))
    # Random starts
    for k in range(4):
        inits.append(_random_points(N, rng))
    
    best_pts = None
    best_energy = 1e30
    
    # First pass: quick repulsion refinement on all inits
    for init in inits:
        if time.time() > deadline:
            break
        refined = _repulsion_refine(init, n_iters=500, lr=0.005, deadline=deadline)
        e = _log_energy_vec(refined)
        if e < best_energy:
            best_energy = e
            best_pts = refined.copy()
    
    # Second pass: gradient-based optimization on best candidates
    # Use scipy minimize with the vectorized energy
    for init in inits[:4]:  # top few initial configs
        if time.time() > deadline:
            break
        # Use the repulsion-refined version as starting point
        refined = _repulsion_refine(init, n_iters=300, lr=0.005, deadline=deadline)
        
        def fun_and_grad(x):
            pts = x.reshape(-1, 3)
            # Project to sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            e, g = _log_energy_grad(pts)
            return e, g.ravel()
        
        x0 = refined.ravel()
        try:
            res = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'gtol': 1e-10, 'maxfun': 1000})
            if res.success or res.fun < best_energy:
                pts = res.x.reshape(-1, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                pts = pts / norms
                e = _log_energy_vec(pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Final: more aggressive refinement on the best
    if best_pts is not None and time.time() < deadline:
        refined = _repulsion_refine(best_pts, n_iters=2000, lr=0.002, deadline=deadline)
        e = _log_energy_vec(refined)
        if e < best_energy:
            best_energy = e
            best_pts = refined.copy()
        
        # Try gradient-based again
        if time.time() < deadline:
            def fun_and_grad(x):
                pts = x.reshape(-1, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                pts = pts / norms
                e, g = _log_energy_grad(pts)
                return e, g.ravel()
            
            x0 = best_pts.ravel()
            try:
                res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                              options={'maxiter': 1000, 'gtol': 1e-12, 'maxfun': 2000})
                pts = res.x.reshape(-1, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                pts = pts / norms
                e = _log_energy_vec(pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = pts.copy()
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
