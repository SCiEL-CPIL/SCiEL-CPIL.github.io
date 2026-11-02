"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_polys(x):
    """Compute H0, H4, H8, H12 at points x (physicists' Hermite polynomials)."""
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2 * x2 - 6 * x2 + 3
    H8 = x2**4 - 28 * x2**3 + 210 * x2**2 - 420 * x2 + 105
    H12 = x2**6 - 60 * x2**5 + 1260 * x2**4 - 12600 * x2**3 + 56700 * x2**2 - 90720 * x2 + 10395
    return H0, H4, H8, H12


def _compute_bound(a0, a4, a8):
    """Compute the upper bound on C4 given coefficients a0, a4, a8.
    
    a12 is determined by g(0) = 0 constraint.
    Uses FFT to compute the Fourier transform numerically.
    """
    a12 = -(a0 + 3 * a4 + 105 * a8) / 10395.0
    
    # Use a fine grid for accurate integration
    x = np.linspace(-25, 25, 8001)
    dx = x[1] - x[0]
    
    H0, H4, H8, H12 = _hermite_polys(x)
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    poly_g = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly_g * np.exp(-x**2 / 2)
    
    # L2 norm squared
    norm_g_sq = np.sum(g**2) * dx
    
    if norm_g_sq < 1e-15:
        return 1e10
    
    # Compute Fourier transform numerically using FFT
    N_fft = 16384
    x_fft = np.linspace(-25, 25, N_fft, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    
    H0f, H4f, H8f, H12f = _hermite_polys(x_fft)
    poly_f = a0 * H0f + a4 * H4f + a8 * H8f + a12 * H12f
    g_fft = poly_f * np.exp(-x_fft**2 / 2)
    
    # Fourier transform
    G = np.fft.fftshift(np.fft.fft(g_fft)) * dx_fft
    k = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dx_fft)) * 2 * np.pi
    
    # L2 norm of Fourier transform
    norm_ghat_sq = np.sum(np.abs(G)**2) * dx_fft
    
    # Compute moments
    x2_g = np.sum(x**2 * g**2) * dx
    k2_ghat = np.sum(k**2 * np.abs(G)**2) * dx_fft
    
    var_x = x2_g / norm_g_sq
    var_k = k2_ghat / norm_ghat_sq if norm_ghat_sq > 1e-15 else 1e10
    
    # The uncertainty product bound
    bound = var_x * var_k
    
    return bound


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180  # 180 second limit
    
    # Initial guess from prior state of the art
    best_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_bound = _compute_bound(*best_result)
    
    # Try to improve with numerical optimization
    try:
        def objective(params):
            return _compute_bound(params[0], params[1], params[2])
        
        # Use Nelder-Mead for robustness
        result = minimize(
            objective,
            best_result,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.fun < best_bound:
            best_result = result.x
            best_bound = result.fun
        
        # Check time and try more iterations if possible
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            result2 = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result2.fun < best_bound:
                best_result = result2.x
                best_bound = result2.fun
        
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
