"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

_start_time = time.time()
_TIME_LIMIT = 180.0  # safety margin below 210s kill

def _hermite(n, x):
    """Probabilists' Hermite polynomial evaluated at x."""
    return eval_hermite(n, x)

def _build_g(coeffs, x):
    """Build g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2).
    c12 is determined by g(0) = 0."""
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    c0, c4, c8 = coeffs
    # At x=0: H0(0)=1, H4(0)=-12, H8(0)=1008, H12(0)=-10395
    # g(0) = (c0*1 + c4*(-12) + c8*1008 + c12*(-10395)) * 1 = 0
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1008.0
    H12_0 = -10395.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = poly * np.exp(-x**2 / 2.0)
    return g

def _compute_uncertainty_product(coeffs, n_points=100000, x_max=15.0):
    """Compute the uncertainty product A(f)*A(ĝ) numerically."""
    x = np.linspace(-x_max, x_max, n_points)
    dx = x[1] - x[0]
    
    g = _build_g(coeffs, x)
    
    # Normalize g so that integral |g|^2 = 1
    norm_sq = np.trapz(g**2, x)
    if norm_sq < 1e-15:
        return 1e10
    g = g / np.sqrt(norm_sq)
    
    # A(g) = sqrt( integral x^2 |g|^2 dx )
    A_g_sq = np.trapz(x**2 * g**2, x)
    A_g = np.sqrt(max(A_g_sq, 0))
    
    # Fourier transform: use FFT or direct
    # For Fourier-invariant setup, use FFT
    # Resample to uniform grid suitable for FFT
    n_fft = 2**18  # 262144
    x_fft = np.linspace(-x_max, x_max, n_fft)
    dx_fft = x_fft[1] - x_fft[0]
    
    g_fft = _build_g(coeffs, x_fft)
    norm_sq_fft = np.trapz(g_fft**2, x_fft)
    if norm_sq_fft < 1e-15:
        return 1e10
    g_fft = g_fft / np.sqrt(norm_sq_fft)
    
    # FFT: Fourier transform with convention F(f)(k) = int f(x) exp(-2πikx) dx
    # numpy fft uses exp(-2πi kn/N), so we need to scale
    G = np.fft.fft(g_fft) * dx_fft
    # Frequency axis
    k = np.fft.fftfreq(n_fft, d=dx_fft)
    
    # Reorder to centered
    G_centered = np.fft.fftshift(G)
    k_centered = np.fft.fftshift(k)
    
    # A(ĝ) = sqrt( integral k^2 |ĝ(k)|^2 dk )
    A_G_sq = np.trapz(k_centered**2 * np.abs(G_centered)**2, k_centered)
    A_G = np.sqrt(max(A_G_sq, 0))
    
    return A_g * A_G

def _objective(coeffs):
    """Objective: the uncertainty product (lower is better)."""
    if time.time() - _start_time > _TIME_LIMIT - 5:
        return 1e10
    return _compute_uncertainty_product(coeffs)

def construct():
    """Return three coefficients [c0, c4, c8] for the Hermite combination."""
    deadline = _start_time + _TIME_LIMIT - 10  # safety margin
    
    # Baseline from Goncalves et al. 2017
    baseline = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _compute_uncertainty_product(baseline)
    best_coeffs = baseline.copy()
    
    if time.time() > deadline:
        return best_coeffs
    
    # Try multiple optimization strategies
    # Strategy 1: Local optimization from baseline
    try:
        res = minimize(_objective, baseline, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        if time.time() < deadline and res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Strategy 2: Try from perturbed starting points
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        perturbation = np.random.normal(0, 0.1, size=3)
        start = baseline + perturbation
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Strategy 3: Homotopy continuation - start from simplified problem
    # Simplified: only H0 and H4 (set c8=0), find optimal c0, c4
    # Then gradually introduce c8
    
    # Two-term: c8=0, optimize c0, c4
    def obj_two_term(params):
        if time.time() > deadline:
            return 1e10
        c0, c4 = params
        return _compute_uncertainty_product(np.array([c0, c4, 0.0]))
    
    try:
        # Start with c0=1, c4=0 (pure Gaussian)
        res2 = minimize(obj_two_term, np.array([1.0, 0.0]), method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        c0_opt, c4_opt = res2.x
        if time.time() < deadline:
            # Now introduce c8 gradually
            for c8_frac in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                if time.time() > deadline:
                    break
                start3 = np.array([c0_opt, c4_opt, c8_frac])
                try:
                    res3 = minimize(_objective, start3, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res3.fun < best_score:
                        best_score = res3.fun
                        best_coeffs = res3.x.copy()
                except Exception:
                    pass
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Strategy 4: Final polish with tighter tolerances
    try:
        res_polish = minimize(_objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res_polish.fun < best_score:
            best_score = res_polish.fun
            best_coeffs = res_polish.x.copy()
    except Exception:
        pass
    
    # Return the best coefficients found
    return best_coeffs
# EVOLVE-BLOCK-END
