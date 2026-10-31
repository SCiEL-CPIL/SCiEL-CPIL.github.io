"""Optimized program: analytic-gradient L-BFGS-B with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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

def _compute_bound(coeffs):
    """Compute the bound value and analytic gradient for given coefficients."""
    a0, a4, a8 = coeffs
    
    # Compute a12 from g(0) = 0 constraint
    # a12 = -(a0*1 + a4*(-12) + a8*168) / (-4960) = (a0 - 12*a4 + 168*a8) / 4960
    a12 = (a0 - 12.0 * a4 + 168.0 * a8) / 4960.0
    
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
    
    # Ensure non-negative under sqrt
    I_gp2 = max(I_gp2, 1e-300)
    I_G2 = max(I_G2, 1e-300)
    
    sqrt_gp2 = np.sqrt(I_gp2)
    sqrt_G2 = np.sqrt(I_G2)
    val = abs(I_g) * sqrt_gp2 * sqrt_G2
    
    # Analytic gradient via chain rule
    # val = |I_g| * sqrt(I_gp2) * sqrt(I_G2)
    # dval/dai = sign(I_g) * dI_g/dai * sqrt(I_gp2) * sqrt(G2)
    #           + I_g * (1/(2*sqrt(I_gp2))) * dI_gp2/dai * sqrt(G2)
    #           + I_g * sqrt(I_gp2) * (1/(2*sqrt(I_G2))) * dI_G2/dai
    
    sign_Ig = 1.0 if I_g >= 0 else -1.0
    
    # dI_g/dai: derivative of integral of g w.r.t. each coefficient
    # g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp
    # da12/da0 = 1/4960, da12/da4 = -12/4960, da12/da8 = 168/4960
    da12_da = np.array([1.0, -12.0, 168.0]) / 4960.0
    
    H_all = np.array([_H0, _H4, _H8, _H12])  # shape (4, N)
    dH_all = np.array([_H0p, _H4p, _H8p, _H12p])  # shape (4, N)
    
    # For each coefficient ai, the derivative of poly w.r.t. ai:
    # d(poly)/da_i = H_i + (da12/da_i) * H12
    # d(g)/da_i = (H_i + da12_da[i]*H12) * exp
    # d(gp)/da_i = (dH_i + da12_da[i]*dH12 - x*(H_i + da12_da[i]*H12)) * exp
    
    grad = np.zeros(3)
    for i in range(3):
        d_poly = H_all[i] + da12_da[i] * _H12
        d_g = d_poly * _EXP
        d_pp = dH_all[i] + da12_da[i] * _H12p
        d_gp = (d_pp - _X * d_poly) * _EXP
        
        dI_g = np.trapz(d_g, _X)
        dI_gp2 = np.trapz(2.0 * gp * d_gp, _X)
        
        # FFT of d_g
        Dg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(d_g))) * _DX
        Dg_real = Dg.real[_G_IDX]
        Dg_x = np.interp(_X, _OMEGA_SORTED_SORTED, Dg_real)
        dI_G2 = np.trapz(2.0 * Gx * Dg_x, _X)
        
        grad[i] = (sign_Ig * dI_g * sqrt_gp2 * sqrt_G2
                   + I_g * (dI_gp2 / (2.0 * sqrt_gp2)) * sqrt_G2
                   + I_g * sqrt_gp2 * (dI_G2 / (2.0 * sqrt_G2)))
    
    return val, grad


def _objective_and_grad(coeffs):
    """Return (value, grad) for the upper bound C4 given (a0,a4,a8)."""
    return _compute_bound(coeffs)


def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art as a starting point.
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    try:
        val0, _ = _objective_and_grad(best_coeffs)
        best_val = val0
    except Exception:
        return best_coeffs
    
    # Multi-start L-BFGS-B
    rng = np.random.default_rng(0)
    starts = [best_coeffs.copy()]
    for _ in range(10):
        starts.append(best_coeffs + rng.normal(0, 0.05, size=3))
    
    for x0 in starts:
        if time.time() > _DEADLINE - _MARGIN:
            break
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish pass
    if time.time() < _DEADLINE - _MARGIN:
        try:
            res = minimize(
                _objective_and_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-14},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        return np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    return best_coeffs
# EVOLVE-BLOCK-END
