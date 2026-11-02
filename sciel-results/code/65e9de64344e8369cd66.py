"""Rewritten program: optimize the uncertainty bound for Hermite-Gaussian test functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 205.0  # safety margin before 210s kill


def _H(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    # Use recurrence to avoid scipy dependency issues
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    Hn_minus_2 = np.ones_like(x)
    Hn_minus_1 = 2 * x
    for k in range(2, n + 1):
        Hn = 2 * x * Hn_minus_1 - 2 * (k - 1) * Hn_minus_2
        Hn_minus_2 = Hn_minus_1
        Hn_minus_1 = Hn
    return Hn_minus_1


def _g_and_ghat(coeffs, x):
    """Compute g(x) and g_hat(xi) at given points.
    g(x) = sum a_n H_n(x) exp(-x^2/2)
    g_hat(xi) = sum a_n (-i)^n H_n(xi) exp(-xi^2/2)  [for this convention]
    For even n, (-i)^n = 1.
    d is determined by g(0) = 0.
    H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-20321280
    """
    a, b, c = coeffs
    d = (a - 12 * b + 1680 * c) / 20321280.0
    
    exp_term = np.exp(-x**2 / 2)
    g = (a * _H(0, x) + b * _H(4, x) + c * _H(8, x) + d * _H(12, x)) * exp_term
    
    # For even n, (-i)^n = 1
    g_hat = (a * _H(0, x) + b * _H(4, x) + c * _H(8, x) + d * _H(12, x)) * exp_term
    return g, g_hat


def _bound(coeffs):
    """Compute the uncertainty bound C4 = ||g||_1 / min|g_hat|."""
    if time.time() > _DEADLINE:
        return np.inf
    
    # Normalize to unit L2
    x_grid = np.linspace(-15, 15, 5000)
    g, g_hat = _g_and_ghat(coeffs, x_grid)
    l2_sq = np.trapz(g**2, x_grid)
    if l2_sq < 1e-20:
        return np.inf
    l2 = np.sqrt(l2_sq)
    g = g / l2
    g_hat = g_hat / l2  # FT is unitary up to constants
    
    l1 = np.trapz(np.abs(g), x_grid)
    
    # Find min |g_hat| more carefully
    min_abs = np.min(np.abs(g_hat))
    if min_abs < 1e-15:
        return np.inf
    
    return l1 / min_abs


def _fibonacci_sphere_3d(N):
    """N points on unit 2-sphere in R^3."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    z = 1 - 2 * np.arange(N) / (N - 1) if N > 1 else np.array([0.0])
    radius = np.sqrt(np.maximum(0, 1 - z**2))
    phi = golden_angle * np.arange(N)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return np.column_stack([x, y, z])


def _refine_sphere(point, n_iter=500, step_init=0.1):
    """Local search on the sphere."""
    best = point.copy()
    best_val = _bound(best)
    step = step_init
    
    for _ in range(n_iter):
        if time.time() > _DEADLINE - 10:
            break
        improved = False
        for _ in range(30):
            delta = np.random.randn(3) * step
            new_p = best + delta
            nrm = np.linalg.norm(new_p)
            if nrm < 1e-15:
                continue
            new_p /= nrm
            val = _bound(new_p)
            if val < best_val - 1e-12:
                best = new_p
                best_val = val
                improved = True
        if not improved:
            step *= 0.9
        if step < 1e-6:
            break
    
    return best, best_val


def construct():
    """Return three coefficients for H0, H4, H8."""
    np.random.seed(12345)
    
    # Seed point (normalized)
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_norm = np.linalg.norm(seed)
    seed_unit = seed / seed_norm
    
    best_point = seed_unit.copy()
    best_val = _bound(best_point)
    
    # Coarse Fibonacci search
    N = 3000
    points = _fibonacci_sphere_3d(N)
    
    for i in range(N):
        if time.time() > _DEADLINE - 60:
            break
        p = points[i]
        val = _bound(p)
        if val < best_val:
            best_val = val
            best_point = p.copy()
    
    # Local refinement
    if time.time() < _DEADLINE - 30:
        best_point, best_val = _refine_sphere(best_point, n_iter=400, step_init=0.08)
    
    # Try from seed too
    if time.time() < _DEADLINE - 30:
        p2, v2 = _refine_sphere(seed_unit, n_iter=300, step_init=0.05)
        if v2 < best_val:
            best_point = p2
            best_val = v2
    
    # Return coefficients. Scale to a reasonable magnitude.
    # The bound is scale-invariant, but let's return with L2 norm = 1 for cleanliness.
    result = best_point.copy()
    return result
# EVOLVE-BLOCK-END
