"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e10  # penalty for coincident points
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _flatten(pts):
    return pts.flatten()


def _unflatten(x):
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _objective(x):
    pts = _unflatten(x)
    return _log_energy(pts)


def _gradient(x):
    """Numerical gradient via central differences."""
    pts = _unflatten(x)
    n = len(pts)
    grad = np.zeros_like(x)
    eps = 1e-6
    base = _objective(x)
    for k in range(len(x)):
        x_plus = x.copy()
        x_plus[k] += eps
        x_minus = x.copy()
        x_minus[k] -= eps
        grad[k] = (_objective(x_plus) - _objective(x_minus)) / (2 * eps)
    return grad


def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = _flatten(pts0)
    
    # Use L-BFGS-B with numerical gradient
    # Since the problem is on the sphere, we optimize in R^3 and renormalize
    # L-BFGS-B works well for smooth objectives
    
    result = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        jac=_gradient,
        options={
            'maxiter': 5000,
            'ftol': 1e-15,
            'gtol': 1e-10,
            'maxls': 50
        }
    )
    
    pts_opt = _unflatten(result.x)
    
    # Ensure all points are on the unit sphere and valid
    norms = np.linalg.norm(pts_opt, axis=1)
    pts_opt = pts_opt / norms[:, np.newaxis]
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts_opt[i] - pts_opt[j])
            if d < 1e-9:
                # Perturb if too close
                pts_opt[j] += 1e-8 * np.random.randn(3)
                pts_opt[j] /= np.linalg.norm(pts_opt[j])
    
    # Final normalization
    norms = np.linalg.norm(pts_opt, axis=1)
    pts_opt = pts_opt / norms[:, np.newaxis]
    
    return pts_opt
# EVOLVE-BLOCK-END
