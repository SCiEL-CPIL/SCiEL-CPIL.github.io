"""Optimized program: exploit homogeneity to reduce search dimension, use grid + Brent."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 5.0

# Precompute Hermite polynomial values on a fixed grid
_NPTS = 2001
_XMIN = -12.0
_XMAX = 12.0
_X = np.linspace(_XMIN, _XMAX, _NPTS)
_DX = _X[1] - _X[0]
_EXP = np.exp(-0.5 * _X * _X)

def _hermite(n, x):
    """Physicist's Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        h_nm2 = np.ones_like(x)
        h_nm1 = 2.0 * x
        for k in range(2, n + 1):
            h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
            h_nm2, h_nm1 = h_nm1, h_n
        return h_nm1

# Precompute Hermite polynomials and their derivatives
_H0 = _hermite(0, _X)
_H4 = _hermite(4, _X)
_H8 = _hermite(8, _X)
_H12 = _hermite(12, _X)
_H3 = _hermite(3, _X)
_H7 = _hermite(7, _X)
_H11 = _hermite(11, _X)

# Derivatives: H_n'(x) = 2n H_{n-1}(x)
_H0p = np.zeros_like(_X)
_H4p = 8.0 * _H3
_H8p = 16.0 * _H7
_H12p = 24.0 * _H11

# Fourier transform kernel (precompute FFT setup)
_N = _NPTS
_OMEGA = 2.0 * np.pi * np.fft.fftfreq(_N, d=_DX)
_OMEGA_SORTED = np.fft.fftshift(_OMEGA)
_G_IDX = np.argsort(_OMEGA_SORTED)
_OMEGA_SORTED_SORTED = _OMEGA_SORTED[_G_IDX]

def _compute_bound_from_ratios(r4, r8):
    """Compute the bound value for given ratios a4/a0 and a8/a0, with a0=1."""
    a0 = 1.0
    a4 = r4
    a8 = r8
    
    # Compute a12 from g(0) = 0 constraint
    a12 = -(a0 * 1.0 + a4 * (-12.0) + a8 * 168.0) / (-4960.0)
    
    # Build g(x) = p(x) * exp(-x^2/2)
    poly = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = poly * _EXP
    
    # Compute g'(x) = (p'(x) - x*p(x)) * exp(-x^2/2)
    pp = a0 * _H0p + a4 * _H4p + a8 * _H8p + a12 * _H12p
    gp = (pp - _X * poly) * _EXP
    
    # Fourier transform via FFT
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * _DX
    G_real = G.real[_G_IDX]
    
    # Interpolate to match _X grid
    Gx = np.interp(_X, _OMEGA_SORTED_SORTED, G_real)
    
    # Compute integrals using trapezoidal rule
    I_g = np.trapz(g, _X)
    I_gp2 = np.trapz(gp * gp, _X)
    I_G2 = np.trapz(Gx * Gx, _X)
    
    val = abs(I_g) * np.sqrt(max(I_gp2, 0.0)) * np.sqrt(max(I_G2, 0.0))
    
    return val

def _compute_bound(coeffs):
    """Compute the bound value for given coefficients (a0, a4, a8)."""
    a0, a4, a8 = coeffs
    if abs(a0) < 1e-10:
        return 1e10  # Degenerate case
    r4 = a4 / a0
    r8 = a8 / a0
    return _compute_bound_from_ratios(r4, r8)

def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art as a starting point
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    try:
        val0 = _compute_bound(best_coeffs)
        best_val = val0
    except Exception:
        return best_coeffs
    
    # Strategy: exploit homogeneity to reduce to 2D search over (r4, r8) = (a4/a0, a8/a0)
    # Use a grid + local refinement approach
    
    # Phase 1: Coarse grid search over reasonable ranges
    # Based on the seed solution: a0=-1.13, a4=0.04, a8=0.000309
    # So r4 ≈ -0.0354, r8 ≈ -0.000273
    # Search in a range around these values
    
    r4_range = np.linspace(-0.1, 0.1, 41)
    r8_range = np.linspace(-0.005, 0.005, 41)
    
    best_r4 = r4_range[0]
    best_r8 = r8_range[0]
    best_val_grid = float('inf')
    
    for r4 in r4_range:
        if time.time() > _DEADLINE - _MARGIN:
            break
        for r8 in r8_range:
            try:
                val = _compute_bound_from_ratios(r4, r8)
                if val < best_val_grid:
                    best_val_grid = val
                    best_r4 = r4
                    best_r8 = r8
            except Exception:
                continue
    
    # Phase 2: Local refinement using Brent's method in 1D slices
    # For fixed r4, optimize over r8, and vice versa
    
    best_r4_final = best_r4
    best_r8_final = best_r8
    best_val_final = best_val_grid
    
    # Refine around the best grid point
    r4_center = best_r4
    r8_center = best_r8
    
    # Try several local windows
    for window_r4 in [0.01, 0.005, 0.002]:
        if time.time() > _DEADLINE - _MARGIN:
            break
        for window_r8 in [0.001, 0.0005, 0.0002]:
            if time.time() > _DEADLINE - _MARGIN:
                break
            
            # Optimize over r4 with r8 fixed at r8_center
            def obj_r4(r4_val):
                return _compute_bound_from_ratios(r4_val, r8_center)
            
            try:
                res_r4 = minimize_scalar(
                    obj_r4,
                    bounds=(r4_center - window_r4, r4_center + window_r4),
                    method='bounded',
                    options={'xatol': 1e-8}
                )
                if res_r4.success and res_r4.fun < best_val_final:
                    best_r4_final = res_r4.x
                    best_val_final = res_r4.fun
            except Exception:
                pass
            
            # Optimize over r8 with r4 fixed at best_r4_final
            def obj_r8(r8_val):
                return _compute_bound_from_ratios(best_r4_final, r8_val)
            
            try:
                res_r8 = minimize_scalar(
                    obj_r8,
                    bounds=(r8_center - window_r8, r8_center + window_r8),
                    method='bounded',
                    options={'xatol': 1e-10}
                )
                if res_r8.success and res_r8.fun < best_val_final:
                    best_r8_final = res_r8.x
                    best_val_final = res_r8.fun
            except Exception:
                pass
            
            # Iterate a few times to converge
            for _ in range(3):
                if time.time() > _DEADLINE - _MARGIN:
                    break
                
                def obj_r4_iter(r4_val):
                    return _compute_bound_from_ratios(r4_val, best_r8_final)
                
                try:
                    res_r4 = minimize_scalar(
                        obj_r4_iter,
                        bounds=(best_r4_final - window_r4, best_r4_final + window_r4),
                        method='bounded',
                        options={'xatol': 1e-8}
                    )
                    if res_r4.success and res_r4.fun < best_val_final:
                        best_r4_final = res_r4.x
                        best_val_final = res_r4.fun
                except Exception:
                    pass
                
                def obj_r8_iter(r8_val):
                    return _compute_bound_from_ratios(best_r4_final, r8_val)
                
                try:
                    res_r8 = minimize_scalar(
                        obj_r8_iter,
                        bounds=(best_r8_final - window_r8, best_r8_final + window_r8),
                        method='bounded',
                        options={'xatol': 1e-10}
                    )
                    if res_r8.success and res_r8.fun < best_val_final:
                        best_r8_final = res_r8.x
                        best_val_final = res_r8.fun
                except Exception:
                    pass
    
    # Convert back to coefficients: a0=1, a4=r4, a8=r8
    # But we need to match the scale of the original solution
    # The seed has a0=-1.13, so let's use that scale
    scale = -113.0 / 100
    result_coeffs = np.array([scale, scale * best_r4_final, scale * best_r8_final])
    
    # Safety: ensure finite and compare with original
    if np.all(np.isfinite(result_coeffs)):
        result_val = _compute_bound(result_coeffs)
        if result_val < best_val:
            return result_coeffs
        else:
            return best_coeffs
    else:
        return best_coeffs
# EVOLVE-BLOCK-END
