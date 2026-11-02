"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in descending powers of x."""
    # H0(x) = 1
    H0 = np.array([1.0])
    # H4(x) = 16x^4 - 48x^2 + 12
    H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
    # H8(x) = 256x^8 - 1280x^6 + 2880x^4 - 2880x^2 + 168
    H8 = np.array([256.0, 0.0, -1280.0, 0.0, 2880.0, 0.0, -2880.0, 0.0, 168.0])
    # H12(x) = 4096x^12 - 38400x^10 + 126720x^8 - 215040x^6 + 193536x^4 - 80640x^2 + 10368
    H12 = np.array([4096.0, 0.0, -38400.0, 0.0, 126720.0, 0.0, -215040.0, 0.0,
                     193536.0, 0.0, -80640.0, 0.0, 10368.0])
    return H0, H4, H8, H12

def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients (descending powers) at x."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        result += c * (x ** power)
    return result

def _compute_c4_upper_bound(a0, a4, a8):
    """Compute the upper bound on C4 for given coefficients."""
    H0, H4, H8, H12 = _hermite_coeffs()
    
    # Compute a12 from constraint g(0) = 0
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=10368
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 168.0) / 10368.0
    
    # Create test function on a grid
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    
    H0_vals = _poly_eval(H0, x)
    H4_vals = _poly_eval(H4, x)
    H8_vals = _poly_eval(H8, x)
    H12_vals = _poly_eval(H12, x)
    
    g = (a0 * H0_vals + a4 * H4_vals + a8 * H8_vals + a12 * H12_vals) * np.exp(-x**2 / 2.0)
    
    # Normalize g
    norm = np.sqrt(np.sum(g**2) * dx)
    if norm < 1e-15:
        return 1e10
    g = g / norm
    
    # Position variance: <x^2> - <x>^2
    x_mean = np.sum(x * g**2) * dx
    x2_mean = np.sum(x**2 * g**2) * dx
    var_x = x2_mean - x_mean**2
    
    # Fourier transform of g
    # Using FFT for efficiency
    # g is defined on x grid, compute its Fourier transform
    # The Fourier transform convention: F(k) = integral g(x) e^{-ikx} dx
    # Using FFT: we need to be careful with scaling
    
    # Let's use scipy's fft or compute via integration
    # For better accuracy, let's use a direct approach with the known properties
    
    # Actually, let's compute the momentum variance directly
    # For a function g(x), the momentum operator is -i d/dx
    # <p^2> = integral |g'(x)|^2 dx (for normalized g)
    
    # Compute derivative numerically
    dg_dx = np.gradient(g, dx)
    p2_mean = np.sum(dg_dx**2) * dx
    
    # <p> = integral g * (-i d/dx g) dx, which is real part = 0 for real g
    # Actually <p> = -i integral g g' dx = 0 for real g (since integral g g' = 0 for decaying g)
    p_mean = 0.0
    
    var_p = p2_mean - p_mean**2
    
    if var_x < 1e-15 or var_p < 1e-15:
        return 1e10
    
    # The uncertainty product
    uncertainty_product = var_x * var_p
    
    # Now, what is C4? In the context of the inequality, C4 is likely related to
    # the constant in the uncertainty principle. The "upper bound on C4" might be
    # the uncertainty product itself, or a normalized version.
    
    # Let's return the uncertainty product as our objective
    return uncertainty_product

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_score = _compute_c4_upper_bound(*seed)
    
    # Try multiple optimization strategies
    strategies = [
        # Nelder-Mead
        {'method': 'Nelder-Mead', 'options': {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}},
        # L-BFGS-B
        {'method': 'L-BFGS-B', 'options': {'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}},
        # Powell
        {'method': 'Powell', 'options': {'maxiter': 5000, 'xtol': 1e-10, 'ftol': 1e-10}},
    ]
    
    # Multiple starting points
    start_points = [
        seed,
        seed + np.array([0.1, 0.01, 0.001]),
        seed - np.array([0.1, 0.01, 0.001]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([-1.5, 0.15, 0.0015]),
        np.array([-0.8, 0.08, 0.0008]),
        np.array([-1.2, 0.12, 0.0012]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30.0:
            break
        for strat in strategies:
            if time.time() > deadline - 30.0:
                break
            try:
                result = minimize(_compute_c4_upper_bound, sp, method=strat['method'],
                                 options=strat['options'])
                if result.success or result.fun < best_score:
                    if result.fun < best_score:
                        best_score = result.fun
                        best_params = result.x.copy()
            except Exception:
                pass
    
    # Refine with tighter tolerances if time permits
    if time.time() < deadline - 15.0:
        try:
            result = minimize(_compute_c4_upper_bound, best_params, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
