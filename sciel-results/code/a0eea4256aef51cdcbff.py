"""Minimize the uncertainty upper bound C4 for a Hermite-Gaussian test function."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite

# Deadline guard: kill before 210s wall-clock, keep safety margin.
_START = time.time()
_DEADLINE = 200.0  # 10s margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _a12_from_p0(a0, a4, a8):
    """Derive a12 so that g(0) = 0."""
    # H_n(0) for even n: H_0(0)=1, H_4(0)=-3, H_8(0)=105, H_12(0)=-10395
    # g(0) = a0*1 + a4*(-3) + a8*105 + a12*(-10395) = 0
    return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0


def _build_g(a0, a4, a8, x):
    """Build g(x) = (a0 H_0 + a4 H_4 + a8 H_8 + a12 H_12) * exp(-x^2/2)."""
    a12 = _a12_from_p0(a0, a4, a8)
    gauss = np.exp(-0.5 * x * x)
    return (a0 * eval_hermite(0, x) + a4 * eval_hermite(4, x)
            + a8 * eval_hermite(8, x) + a12 * eval_hermite(12, x)) * gauss


def _c4_bound(a0, a4, a8):
    """Compute C4 = ||g||_1 * ||g_hat||_1 with ||g||_2 = 1 normalization."""
    N = 4096
    x = np.linspace(-12.0, 12.0, N)
    dx = x[1] - x[0]
    
    g = _build_g(a0, a4, a8, x)
    
    # Normalize so ||g||_2 = 1
    norm2 = np.sqrt(np.sum(g * g) * dx)
    if norm2 < 1e-15:
        return 1e10
    g = g / norm2

    # L1 norm of g
    l1_g = np.sum(np.abs(g)) * dx

    # Fourier transform via FFT
    gft = np.fft.fftshift(np.fft.fft(g))
    # Unitary FFT scaling
    gft = gft * (dx / np.sqrt(2.0 * np.pi))
    l1_gft = np.sum(np.abs(gft)) * (2.0 * np.pi / N / dx)

    return l1_g * l1_gft


def _c4_bound_fast(a0, a4, a8):
    """Faster version with smaller grid for initial search."""
    N = 1024
    x = np.linspace(-10.0, 10.0, N)
    dx = x[1] - x[0]
    
    g = _build_g(a0, a4, a8, x)
    
    norm2 = np.sqrt(np.sum(g * g) * dx)
    if norm2 < 1e-15:
        return 1e10
    g = g / norm2

    l1_g = np.sum(np.abs(g)) * dx

    gft = np.fft.fftshift(np.fft.fft(g))
    gft = gft * (dx / np.sqrt(2.0 * np.pi))
    l1_gft = np.sum(np.abs(gft)) * (2.0 * np.pi / N / dx)

    return l1_g * l1_gft


def _objective(params):
    """Objective function with deadline check."""
    if _time_left() < 2.0:
        return 1e10
    try:
        a0, a4, a8 = params
        return _c4_bound_fast(a0, a4, a8)
    except Exception:
        return 1e10


def construct():
    # Initial best from seed
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _c4_bound(*best_params)
    
    # Multi-start local optimization first (fast, exploits smooth landscape)
    start_points = [
        best_params.copy(),
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.8, 0.04, 0.002]),
        np.array([-1.1, 0.03, 0.0005]),
        np.array([-0.9, 0.06, 0.003]),
    ]
    
    for sp in start_points:
        if _time_left() < 10.0:
            break
        try:
            res = minimize(_c4_bound_fast, sp, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_val:
                # Verify with higher-precision computation
                val_full = _c4_bound(*res.x)
                if val_full < best_val:
                    best_val = val_full
                    best_params = res.x.copy()
        except Exception:
            continue
    
    # Differential evolution for global search if time permits
    if _time_left() > 30.0:
        bounds = [(-2.0, 0.5), (-0.5, 0.5), (-0.1, 0.1)]
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=30,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=False
            )
            
            if de_result.fun < best_val:
                # Verify with higher-precision computation
                val_full = _c4_bound(*de_result.x)
                if val_full < best_val:
                    best_val = val_full
                    best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Final refinement with local optimization on best found
    if _time_left() > 15.0:
        try:
            res = minimize(_c4_bound, best_params, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # One more polish if time allows
    if _time_left() > 5.0:
        try:
            res = minimize(_c4_bound, best_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
