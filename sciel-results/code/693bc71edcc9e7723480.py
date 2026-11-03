import numpy as np
from scipy import optimize
import time

_start = time.time()
_DEADLINE = 200.0

def _hermite(n, x):
    """Physicists' Hermite polynomials: H_0=1, H_1=2x, H_2=4x^2-2, ..."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2*x
    H0 = np.ones_like(x)
    H1 = 2*x
    for k in range(1, n):
        H2 = 2*x*H1 - 2*k*H0
        H0, H1 = H1, H2
    return H1

# Precompute with exp(-pi*x^2) Gaussian envelope
_N = 20000
_X = np.linspace(-10, 10, _N)
_DX = _X[1] - _X[0]
_H0 = _hermite(0, _X)
_H4 = _hermite(4, _X)
_H8 = _hermite(8, _X)
_H12 = _hermite(12, _X)
_EXP = np.exp(-np.pi * _X**2)
_X4 = _X**4

# For Fourier transform with exp(-pi*x^2), use:
# FT{H_n(x) exp(-pi x^2)} = (-i)^n H_n(x) exp(-pi x^2)
# So we can compute I2 directly without FFT
_XI4 = _X**4  # same variable name since |xi|^4 = x^4 for the transformed function

def _objective(coeffs):
    """Compute max(I1, I2) where I1 = int x^4 g(x)^2 dx, I2 = int xi^4 |g_hat(xi)|^2 dxi"""
    a0, a4, a8 = coeffs
    # H_n(0) values for physicists' Hermite polynomials
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    
    # g(x) = P(x) * exp(-pi x^2)
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    g = P * _EXP
    
    # I1 = int x^4 g(x)^2 dx
    I1 = np.sum(_X4 * g**2) * _DX
    
    # For g(x) = P(x)exp(-pi x^2), the Fourier transform is:
    # g_hat(xi) = P(-i*xi) * exp(-pi xi^2)  [up to constant, but since we use same grid, it works]
    # Actually, with the convention FT{f}(xi) = int f(x) exp(-2pi i x xi) dx,
    # FT{exp(-pi x^2)} = exp(-pi xi^2)
    # FT{H_n(x) exp(-pi x^2)} = (-i)^n H_n(xi) exp(-pi xi^2)
    # So |g_hat(xi)|^2 = |P(-i*xi)|^2 * exp(-2pi xi^2)
    # But P has real coefficients, so P(-i*xi) is complex in general.
    # However, since H_n has parity, and our combination is even (H0, H4, H8, H12 are all even),
    # P(-i*xi) = a0*H0(-i*xi) + a4*H4(-i*xi) + a8*H8(-i*xi) + a12*H12(-i*xi)
    # H_n(-i*xi) = (-i)^n H_n(xi) for these even n, so:
    # P(-i*xi) = a0*H0(xi) + a4*(-i)^4 H4(xi) + a8*(-i)^8 H8(xi) + a12*(-i)^12 H12(xi)
    # = a0*H0(xi) + a4*H4(xi) + a8*H8(xi) + a12*H12(xi) = P(xi)
    # So |g_hat(xi)|^2 = P(xi)^2 * exp(-2pi xi^2)
    
    g_hat_abs_sq = P**2 * np.exp(-2*np.pi * _X**2)
    I2 = np.sum(_XI4 * g_hat_abs_sq) * _DX
    
    return max(I1, I2)

def construct():
    global _start, _DEADLINE
    
    seed = np.array([-113/100, 1/25, 1/3240], dtype=float)
    best_val = _objective(seed)
    best_x = seed.copy()
    
    rng = np.random.default_rng(12345)
    starts = [seed.copy()]
    # Add perturbations around seed with various scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        for _ in range(8):
            starts.append(seed + scale * rng.standard_normal(3))
    # Random starts in a reasonable range
    for _ in range(15):
        starts.append(rng.uniform(-5, 5, size=3))
    
    # First pass: Nelder-Mead from multiple starts
    for x0 in starts:
        if time.time() - _start > _DEADLINE - 30:
            break
        try:
            res = optimize.minimize(_objective, x0, method='Nelder-Mead',
                                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Second pass: L-BFGS-B from the best found and nearby points
    if time.time() - _start < _DEADLINE - 20:
        local_starts = [best_x.copy()]
        for scale in [0.01, 0.05, 0.1]:
            for _ in range(10):
                local_starts.append(best_x + scale * rng.standard_normal(3))
        
        for x0 in local_starts:
            if time.time() - _start > _DEADLINE - 15:
                break
            try:
                res = optimize.minimize(_objective, x0, method='L-BFGS-B',
                                        options={'maxiter': 500, 'gtol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    return np.array(best_x, dtype=float)
