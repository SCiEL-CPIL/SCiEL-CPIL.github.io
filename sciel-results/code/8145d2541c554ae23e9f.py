"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0  # Leave 10s margin before 210s kill

def _hermite_polynomial(n, x):
    """Compute physicist's Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        h0 = np.ones_like(x)
        h1 = 2 * x
        for k in range(2, n + 1):
            h2 = 2 * x * h1 - 2 * (k - 1) * h0
            h0, h1 = h1, h2
        return h1

def _gaussian_weight(x):
    """Gaussian weight exp(-x^2 / 2)."""
    return np.exp(-x**2 / 2)

def _hermite_function_coeff(n):
    """Normalization constant for h_n(x) = H_n(x) * exp(-x^2/2) / sqrt(2^n * n! * pi^(1/2))"""
    # h_n is orthonormal: ∫ h_n^2 dx = 1
    # h_n(x) = (1 / sqrt(2^n * n! * sqrt(pi))) * H_n(x) * exp(-x^2/2)
    return 1.0 / np.sqrt(2**n * factorial(n, exact=True) * np.sqrt(np.pi))

def _compute_g_and_fourier(a0, a1, a2):
    """
    Given coefficients a0, a1, a2 for H0, H4, H8 terms,
    compute a3 for H12 so that g(0) = 0, then compute g(x) and its Fourier transform.
    
    g(x) = (a0*H0(x) + a1*H4(x) + a2*H8(x) + a3*H12(x)) * exp(-x^2/2)
    
    At x=0: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    So: a0*1 + a1*(-3) + a2*105 + a3*(-10395) = 0
    => a3 = (a0 - 3*a1 + 105*a2) / 10395
    """
    a3 = (a0 - 3.0*a1 + 105.0*a2) / 10395.0
    
    # Build g(x) on a fine grid
    x = np.linspace(-15, 15, 20000)
    H0 = _hermite_polynomial(0, x)
    H4 = _hermite_polynomial(4, x)
    H8 = _hermite_polynomial(8, x)
    H12 = _hermite_polynomial(12, x)
    
    poly = a0*H0 + a1*H4 + a2*H8 + a3*H12
    g = poly * _gaussian_weight(x)
    
    # Fourier transform: ĝ(ξ) = ∫ g(x) e^{-2πi x ξ} dx
    # But for Hermite functions, the Fourier transform (unitary, with e^{-i x ξ}) gives eigenvalue (-i)^n
    # We need to be careful about Fourier transform convention.
    # 
    # The standard result: If f(x) = Σ c_n h_n(x) where h_n are orthonormal Hermite functions,
    # then F(f)(ξ) = Σ c_n (-i)^n h_n(ξ) where F uses kernel e^{-i x ξ}/sqrt(2π).
    #
    # But A(f) = ∫|f(x)|dx and A(F(f)) depends on the Fourier convention used in the task.
    # 
    # Let me use numerical FFT to be safe.
    # Use FFT with appropriate scaling.
    
    # Actually, let's use the analytical approach since we know the structure.
    # Convert to Hermite function coefficients.
    # g(x) = Σ_n c_n * h_n(x) where h_n(x) = H_n(x)*exp(-x^2/2)/sqrt(2^n n! sqrt(pi))
    # So c_n = a_n * sqrt(2^n n! sqrt(pi)) for n in {0,4,8,12}
    
    c0 = a0 * np.sqrt(2**0 * factorial(0, exact=True) * np.sqrt(np.pi))
    c4 = a1 * np.sqrt(2**4 * factorial(4, exact=True) * np.sqrt(np.pi))
    c8 = a2 * np.sqrt(2**8 * factorial(8, exact=True) * np.sqrt(np.pi))
    c12 = a3 * np.sqrt(2**12 * factorial(12, exact=True) * np.sqrt(np.pi))
    
    # g(x) = c0*h0(x) + c4*h4(x) + c8*h8(x) + c12*h12(x)
    # F(g)(ξ) = c0*1*h0(ξ) + c4*(-1)*h4(ξ) + c8*1*h8(ξ) + c12*(-1)*h12(ξ)
    # (since (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=-1)
    
    # Wait: (-i)^4 = (-i)^2 * (-i)^2 = (-1)*(-1) = 1
    # (-i)^8 = 1
    # (-i)^12 = (-i)^8 * (-i)^4 = 1*1 = 1
    # Actually: (-i)^1 = -i, (-i)^2 = -1, (-i)^3 = i, (-i)^4 = 1, period 4.
    # So (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1. All are 1!
    
    # So F(g)(ξ) = c0*h0(ξ) + c4*h4(ξ) + c8*h8(ξ) + c12*h12(ξ) = g(ξ)
    # The function is invariant under Fourier transform (up to the convention)!
    
    # This means A(g) = A(F(g)) if we use the right convention.
    # But the L1 norm might differ depending on the Fourier transform normalization.
    
    # Let me just compute numerically to be safe.
    
    # Use FFT for the Fourier transform
    # dx = x[1] - x[0]
    dx = x[1] - x[0]
    N = len(x)
    
    # FFT: F(ξ_k) = sum_j g(x_j) * exp(-2πi x_j ξ_k) * dx
    # ξ_k = k / (N*dx) for k = 0, ..., N-1 (with shift)
    
    G = np.fft.fft(g) * dx
    # Shift so zero frequency is in the middle
    G = np.fft.fftshift(G)
    
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # L1 norms
    A_g = np.sum(np.abs(g)) * dx
    A_G = np.sum(np.abs(G)) * (1.0 / (N * dx))  # dξ = 1/(N*dx)
    
    return A_g * A_G, a3

def _objective(coeffs):
    """Objective function to minimize."""
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e10  # Timeout penalty
    a0, a1, a2 = coeffs
    try:
        val, _ = _compute_g_and_fourier(a0, a1, a2)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    """Construct the three coefficients for H0, H4, H8."""
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess from prior art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_val, best_x = _objective(x0), x0.copy()
    
    # Try multiple local optimizations from different starting points
    starting_points = [
        x0,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.01]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.5, 0.15, 0.015]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.05, 0.001]),
        np.array([-2.0, 0.2, 0.02]),
    ]
    
    for sp in starting_points:
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        try:
            res = minimize(_objective, sp, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.success or res.fun < best_val:
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if time.time() - START_TIME < TIME_LIMIT - 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
            res_de = differential_evolution(_objective, bounds, maxiter=100, 
                                           popsize=20, tol=1e-10, seed=42,
                                           polish=False, init='latinhypercube')
            if np.isfinite(res_de.fun) and res_de.fun < best_val:
                best_val = res_de.fun
                best_x = res_de.x
                # Polish with local optimization
                if time.time() - START_TIME < TIME_LIMIT - 20:
                    res_polish = minimize(_objective, best_x, method='Nelder-Mead',
                                       options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
                    if np.isfinite(res_polish.fun) and res_polish.fun < best_val:
                        best_val = res_polish.fun
                        best_x = res_polish.x
        except Exception:
            pass
    
    # Final polish
    if time.time() - START_TIME < TIME_LIMIT - 10:
        try:
            res_final = minimize(_objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17})
            if np.isfinite(res_final.fun) and res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END

