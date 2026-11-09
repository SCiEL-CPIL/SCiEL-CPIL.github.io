"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial values on a grid
_x_grid = np.linspace(-12, 12, 8192)
_dx = _x_grid[1] - _x_grid[0]
# Gaussian envelope: g(x) = P(x) * exp(-x^2/2)
_gauss = np.exp(-0.5 * _x_grid**2)
# Hermite polynomials H0, H4, H8, H12
_H0 = np.ones_like(_x_grid)
_H4 = eval_hermite(4, _x_grid)
_H8 = eval_hermite(8, _x_grid)
_H12 = eval_hermite(12, _x_grid)

# Values at x=0 for enforcing g(0)=0
_h0_0 = 1.0
_h4_0 = eval_hermite(4, 0.0)
_h8_0 = eval_hermite(8, 0.0)
_h12_0 = eval_hermite(12, 0.0)


def _compute_upper_bound(a0, a4, a8):
    """Compute the upper bound C4 on A(f)*A(F(f))."""
    # a12 is determined by g(0) = 0:
    # a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    a12 = -(a0 * _h0_0 + a4 * _h4_0 + a8 * _h8_0) / _h12_0

    # Build g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    poly = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = poly * _gauss

    # Compute norms
    # ||g||_2^2
    norm2_sq = np.sum(g**2) * _dx
    if norm2_sq <= 0:
        return 1e10

    # ||g||_4^4
    norm4_4 = np.sum(g**4) * _dx

    # Fourier transform of g: since g = P(x) * exp(-x^2/2), and F[exp(-x^2/2)] = sqrt(2*pi)*exp(-w^2/2)
    # We use the property that multiplication by Hermite polynomials corresponds to derivatives in Fourier space,
    # but it's easier to just compute numerically via FFT or use the known relation.
    # Actually, for the uncertainty product, we need ||F(g)||_4^4 and ||F(g)||_2^2.
    # By Plancherel: ||F(g)||_2^2 = ||g||_2^2 (with proper normalization).
    # For ||F(g)||_4, we need the actual Fourier transform values.

    # Use the fact that F[exp(-x^2/2) * H_n(x)] can be computed analytically.
    # Alternatively, compute F(g) numerically using FFT on the grid.
    # The grid is [-12, 12] with 8192 points, dx = 24/8191.
    # Frequency grid: w_k = k * (2*pi / (N*dx)) for k = -N/2..N/2-1
    N = len(_x_grid)
    F_g = np.fft.fftshift(np.fft.fft(g)) * _dx
    w_grid = 2 * np.pi * np.fft.fftfreq(N, d=_dx)
    w_grid = np.fft.fftshift(w_grid)

    norm2_F_sq = np.sum(F_g**2) * _dx  # Should equal norm2_sq by Plancherel (up to 2*pi factor)
    norm4_F_4 = np.sum(F_g**4) * _dx

    # The upper bound on A(f)*A(F(f)):
    # A(f) = ||f||_4^2 / ||f||_2  (or similar)
    # The standard bound: A(f) * A(F(f)) <= C * (||f||_4^2/||f||_2) * (||F(f)||_4^2/||F(f)||_2)
    # So C4 = (||f||_4^2/||f||_2) * (||F(f)||_4^2/||F(f)||_2)
    term1 = np.sqrt(norm4_4) / np.sqrt(norm2_sq)
    term2 = np.sqrt(norm4_F_4) / np.sqrt(norm2_F_sq)
    return term1 * term2


def _objective(coeffs):
    """Objective function for optimization."""
    a0, a4, a8 = coeffs
    return _compute_upper_bound(a0, a4, a8)


def construct():
    """Return the three coefficients [a0, a4, a8] minimizing the upper bound."""
    deadline = time.time() + 180.0  # 180s wall-clock limit

    # Start from the known good solution
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = x0.copy()
    best_val = _objective(x0)

    # Try multiple starting points and optimization methods
    start_points = [
        x0,
        x0 + np.array([0.1, 0.01, 0.001]),
        x0 + np.array([-0.1, 0.01, -0.001]),
        x0 + np.array([0.05, -0.005, 0.0005]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.2, 0.04, 0.0004]),
        np.array([-0.9, 0.06, 0.0006]),
    ]

    methods = ['Nelder-Mead', 'Powell', 'BFGS']

    for method in methods:
        if time.time() > deadline - 15.0:
            break
        for sp in start_points:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    _objective, sp, method=method,
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x
            except Exception:
                pass

    # Final polish with tight tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective, best_x, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass

    return np.array(best_x)
# EVOLVE-BLOCK-END
