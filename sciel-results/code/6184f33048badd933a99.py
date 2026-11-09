"""
Uncertainty inequality: tune Hermite-Gaussian test function coefficients to minimize the upper bound on C4.

We parameterize g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is derived from the constraint g(0) = 0.

The score is the upper bound on A(f)*A(Fourier(f)) (or C4-related constant).
We use analytic expressions for norm^2 and fast numerical evaluation for norm_inf.
The bound is (||g||_inf / ||g||_2)^2.

Key structural insight: the Fourier transform of H_n(x)*exp(-x^2/2) for even n
is H_n(x)*exp(-x^2/2) (up to sign, which doesn't affect abs). So ||F(g)||_inf = ||g||_inf
and ||F(g)||_2 = ||g||_2, making the product bound (||g||_inf/||g||_2)^2.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

DEADLINE_FRACTION = 0.88  # safety margin before 210s limit

# Precompute Hermite polynomial values on a grid for fast norm_inf evaluation
_X_GRID = np.linspace(-12, 12, 30000)
_GAUSS = np.exp(-0.5 * _X_GRID**2)
_H0_GRID = np.ones_like(_X_GRID)
_H4_GRID = eval_hermite(4, _X_GRID)
_H8_GRID = eval_hermite(8, _X_GRID)
_H12_GRID = eval_hermite(12, _X_GRID)

# Analytic norms: integral H_n^2 * exp(-x^2) dx = sqrt(pi) * 2^n * n!
_NORM2_COEFFS = np.array([
    np.sqrt(np.pi) * 2**0 * 1,           # n=0
    np.sqrt(np.pi) * 2**4 * 24,           # n=4
    np.sqrt(np.pi) * 2**8 * 40320,        # n=8
    np.sqrt(np.pi) * 2**12 * 479001600    # n=12
])


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound (||g||_inf / ||g||_2)^2 for the Hermite-Gaussian test function.
    a12 is derived from g(0) = 0.
    """
    # Derive a12 from g(0) = 0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # a0 - 3*a4 + 105*a8 - 10395*a12 = 0
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Compute ||g||_2^2 analytically
    coeffs = np.array([a0, a4, a8, a12])
    norm2_sq = np.dot(coeffs**2, _NORM2_COEFFS)
    
    if norm2_sq <= 0:
        return 1e10
    
    norm2 = np.sqrt(norm2_sq)
    
    # Compute ||g||_inf numerically
    poly = a0*_H0_GRID + a4*_H4_GRID + a8*_H8_GRID + a12*_H12_GRID
    g = poly * _GAUSS
    norm_inf = np.max(np.abs(g))
    
    bound = (norm_inf / norm2)**2
    return bound


def _compute_bound_vectorized(coeffs_array):
    """Vectorized bound computation for multiple coefficient sets."""
    a0, a4, a8 = coeffs_array[:, 0], coeffs_array[:, 1], coeffs_array[:, 2]
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Norm2 squared (analytic)
    coeffs = np.column_stack([a0, a4, a8, a12])
    norm2_sq = np.sum(coeffs**2 * _NORM2_COEFFS, axis=1)
    
    valid = norm2_sq > 0
    norm2 = np.sqrt(np.where(valid, norm2_sq, 1.0))
    
    # Norm_inf (numerical) - process in chunks to manage memory
    n = len(a0)
    bounds = np.full(n, 1e10)
    chunk_size = max(1, min(100, n))
    
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a0c, a4c, a8c, a12c = a0[start:end], a4[start:end], a8[start:end], a12[start:end]
        
        # Shape: (chunk, grid_size)
        poly = a0c[:, None]*_H0_GRID[None, :] + a4c[:, None]*_H4_GRID[None, :] + \
               a8c[:, None]*_H8_GRID[None, :] + a12c[:, None]*_H12_GRID[None, :]
        g = poly * _GAUSS[None, :]
        norm_inf = np.max(np.abs(g), axis=1)
        
        bounds[start:end] = (norm_inf / norm2[start:end])**2
        bounds[start:end][~valid[start:end]] = 1e10
    
    return bounds


def construct():
    start_time = time.time()
    deadline = 210.0 * DEADLINE_FRACTION  # ~184.8s
    
    best_val = float('inf')
    best_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])  # seed
    
    # Evaluate seed
    try:
        seed_val = _compute_bound(best_x[0], best_x[1], best_x[2])
        if seed_val < best_val:
            best_val = seed_val
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        return best_x
    
    # Phase 1: Broad exploration using multi-start Nelder-Mead
    # The landscape is smooth; use several diverse starting points
    np.random.seed(42)
    
    # Define diverse starting points
    start_points = [
        best_x.copy(),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([1.0, -0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([2.0, -0.05, 0.0005]),
        np.array([-0.5, 0.2, 0.002]),
        np.array([0.5, -0.2, 0.002]),
        best_x + 0.5 * np.random.randn(3),
        best_x + 1.0 * np.random.randn(3),
    ]
    
    for x0 in start_points:
        if time.time() - start_time > deadline - 25:
            break
        try:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        return best_x
    
    # Phase 2: Local refinement with tighter tolerances around the best point
    try:
        result = minimize(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_x
    
    # Phase 3: Fine-grained local search with smaller perturbations
    np.random.seed(7)
    for i in range(8):
        if time.time() - start_time > deadline - 5:
            break
        scale = 0.01 * (1.0 + 0.3*i)
        x0 = best_x + scale * np.random.randn(3)
        try:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
