"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e10  # penalty for coincident points
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. each point on sphere.
    
    For points on unit sphere, we project gradient to tangent space.
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Project onto tangent space: grad_tan = grad - (grad . x_i) * x_i
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            grad[i] -= d / (dist * dist)
    # Project to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad


def _sphere_constraint(pts):
    """Flatten points for scipy minimize."""
    return pts.flatten()


def _sphere_unflatten(x):
    """Unflatten and normalize back to unit sphere."""
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def construct():
    # Start with Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Optimize using L-BFGS-B with sphere constraint handled by projection
    # We'll use a custom approach: optimize in R^3 and project back to sphere
    
    x0 = pts.flatten()
    
    def objective(x):
        pts = _sphere_unflatten(x)
        return _log_energy(pts)
    
    def gradient(x):
        pts = _sphere_unflatten(x)
        g = _energy_grad(pts)
        return g.flatten()
    
    # Use L-BFGS-B for optimization
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    pts = _sphere_unflatten(result.x)
    
    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
