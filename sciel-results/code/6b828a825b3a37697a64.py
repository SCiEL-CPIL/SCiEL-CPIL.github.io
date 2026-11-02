"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

_DEADLINE = 200.0  # safety margin before 210s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _hermite_coeffs():
    """Return coefficients [c0, c1, ..., cn] for H_n(x) in power basis x^0 + c1*x^1 + ... + cn*x^n."""
    # H_0 = 1
    # H_4 = x^4 - 6x^2 + 3
    # H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 55440x^2 + 10395
    H0 = np.array([1.0])
    H4 = np.array([3.0, 0.0, -6.0, 0.0, 1.0])
    H8 = np.array([105.0, 0.0, -420.0, 0.0, 210.0, 0.0, -28.0, 0.0, 1.0])
    H12 = np.array([10395.0, 0.0, -55440.0, 0.0, 45360.0, 0.0, -12600.0, 0.0, 1260.0, 0.0, -60.0, 0.0, 1.0])
    return H0, H4, H8, H12

def _compute_g_and_fourier(a0, a4, a8):
    """
    Given a0, a4, a8, compute a12 from g(0)=0, then return the polynomial coefficients
    of the combined Hermite polynomial P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12,
    and the Fourier transform polynomial Q(x) of the Gaussian-weighted version.
    
    The function is g(x) = P(x) * exp(-x^2/2).
    The Fourier transform (using convention F(f)(xi) = int f(x) exp(-2*pi*i*xi*x) dx)
    of P(x)*exp(-x^2/2) can be computed. For simplicity, we'll compute numerically.
    """
    H0, H4, H8, H12 = _hermite_coeffs()
    
    # Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0) = 1, H4(0) = 3, H8(0) = 105, H12(0) = 10395
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
    
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Need to add polynomials with different degrees
    max_deg = 12
    P = np.zeros(max_deg + 1)
    for coeff, poly in [(a0, H0), (a4, H4), (a8, H8), (a12, H12)]:
        for i, c in enumerate(poly):
            P[i] += coeff * c
    
    return P, a12

def _l1_norm_gaussian_weighted_poly(P, n_points=2000, x_range=8.0):
    """
    Compute L1 norm of P(x) * exp(-x^2/2) numerically.
    P is given as coefficients [c0, c1, ..., cn] for c0 + c1*x + ... + cn*x^n.
    """
    x = np.linspace(-x_range, x_range, n_points)
    # Evaluate polynomial
    g = np.zeros_like(x)
    for i in range(len(P) - 1, -1, -1):
        g = g * x + P[i]
    g = g * np.exp(-x**2 / 2.0)
    dx = x[1] - x[0] if n_points > 1 else 1.0
    return np.sum(np.abs(g)) * dx

def _fourier_transform_l1(P, n_points=2000, xi_range=4.0):
    """
    Compute L1 norm of Fourier transform of P(x)*exp(-x^2/2).
    Using FFT-based approach for efficiency.
    """
    # We need F(g)(xi) = int g(x) exp(-2*pi*i*xi*x) dx where g(x) = P(x)*exp(-x^2/2)
    # Use numerical integration via FFT
    x = np.linspace(-8.0, 8.0, 4096)
    dx = x[1] - x[0]
    
    # Evaluate P(x)
    g = np.zeros_like(x)
    for i in range(len(P) - 1, -1, -1):
        g = g * x + P[i]
    g = g * np.exp(-x**2 / 2.0)
    
    # FFT to get Fourier transform
    # The FFT computes sum g(x_k) exp(-2*pi*i*k*n/N)
    # We need to scale appropriately
    G = np.fft.fft(g) * dx
    # Shift to center (zero frequency in middle)
    G = np.fft.fftshift(G)
    
    # Frequency axis
    N = len(x)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # L1 norm
    return np.sum(np.abs(G)) * (1.0 / freqs[1] - 1.0 / freqs[0]) if N > 1 else 0.0

def _uncertainty_product(a0, a4, a8):
    """Compute the uncertainty product A(g) * A(F(g))."""
    if _time_left() < 5.0:
        return np.inf
    
    P, a12 = _compute_g_and_fourier(a0, a4, a8)
    
    # L1 norm of g
    Ag = _l1_norm_gaussian_weighted_poly(P, n_points=2000, x_range=8.0)
    
    # L1 norm of Fourier transform
    AF = _fourier_transform_l1(P, n_points=4096)
    
    return Ag * AF

def construct():
    global _start_time
    _start_time = time.time()
    
    # Best known seed values
    best_x = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _uncertainty_product(*best_x)
    
    if best_val == np.inf or not np.isfinite(best_val):
        return best_x
    
    # Differential evolution for global search
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    try:
        if _time_left() > 60.0:
            result_de = differential_evolution(
                _uncertainty_product,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,  # We'll do our own polish
                init='sobol',
                updating='deferred',
                workers=-1
            )
            if result_de.success and result_de.fun < best_val:
                best_x = result_de.x
                best_val = result_de.fun
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    try:
        if _time_left() > 15.0:
            result_nm = minimize(
                _uncertainty_product,
                x0=best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result_nm.success and result_nm.fun < best_val:
                best_x = result_nm.x
                best_val = result_nm.fun
    except Exception:
        pass
    
    # Try a few more local starts from different points if time permits
    try:
        if _time_left() > 10.0:
            rng = np.random.RandomState(123)
            for _ in range(5):
                if _time_left() < 8.0:
                    break
                x_start = rng.uniform(-3, 3, size=3)
                result_nm2 = minimize(
                    _uncertainty_product,
                    x0=x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result_nm2.success and result_nm2.fun < best_val:
                    best_x = result_nm2.x
                    best_val = result_nm2.fun
    except Exception:
        pass
    
    # Final polish with tighter tolerance
    try:
        if _time_left() > 5.0:
            result_polish = minimize(
                _uncertainty_product,
                x0=best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result_polish.success and result_polish.fun < best_val:
                best_x = result_polish.x
                best_val = result_polish.fun
    except Exception:
        pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
