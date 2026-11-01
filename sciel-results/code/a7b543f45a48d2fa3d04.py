"""Rewritten program for the uncertainty inequality tuning task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, Bounds

# Proportional Hermite polynomials (physicists' convention)
# H_n(x) = (-1)^n exp(x^2) d^n/dx^n exp(-x^2)
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 60480x^2 + 945

def H0(x):
    return 1.0

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def H12(x):
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 45360*x**4 - 60480*x**2 + 945

def make_g(c0, c4, c8):
    """Construct g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    with c12 chosen so that g(0) = 0."""
    def g(x):
        c12 = -(c0 * H0(0) + c4 * H4(0) + c8 * H8(0)) / H12(0)
        poly = c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)
        return poly * np.exp(-x**2 / 2.0)
    return g

def compute_bound(c0, c4, c8, n_grid=10000, x_max=15.0):
    """Compute an upper bound on A(f)*A(F(f)) for the given coefficients.
    
    The uncertainty product bound is related to the L1 norms of g and its Fourier transform.
    For g(x) = P(x) exp(-x^2/2), the Fourier transform is also a Hermite-Gaussian combination.
    
    A(g) = integral |g(x)| dx
    A(F(g)) = (1/sqrt(2*pi)) integral |F(g)(k)| dk
    
    The product A(g)*A(F(g)) gives a bound on the uncertainty constant.
    """
    # Grid for numerical integration
    x = np.linspace(-x_max, x_max, n_grid)
    dx = x[1] - x[0]
    
    # Compute g(x)
    g_vals = make_g(c0, c4, c8)(x)
    
    # Compute Fourier transform of g
    # F(g)(k) = integral g(x) exp(-i k x) dx
    # For our purposes, we need |F(g)(k)| integrated over k
    # Since g is real and even (all Hermite polynomials here are even), F(g) is real and even
    # F(g)(k) = 2 * integral_0^inf g(x) cos(k x) dx
    
    # Use FFT-based approach for efficiency
    # Actually, let's use a direct numerical approach with a fine grid
    # The Fourier transform of a function supported on [-x_max, x_max]
    # F(g)(k) = sum g(x_i) exp(-i k x_i) dx
    
    # For better efficiency, use the fact that we can compute the FT at discrete points
    # and then integrate
    
    # Let's use a smaller grid for the FT computation to stay within time limits
    n_k = 5000
    k_max = 2 * np.pi / dx  # Nyquist-like limit
    k = np.linspace(-k_max, k_max, n_k)
    dk = k[1] - k[0]
    
    # Compute FT using matrix multiplication (might be slow for large grids)
    # Better: use FFT if possible, or a smaller grid
    
    # For speed, let's use a coarser approach
    # The key insight: for Hermite-Gaussian functions, the FT has a known form
    # FT of H_n(x) exp(-x^2/2) = i^n H_n(k) exp(-k^2/2) (up to normalization)
    
    # Actually, let's use the analytical property:
    # If g(x) = sum c_n H_n(x) exp(-x^2/2), then
    # F(g)(k) = i^n sum c_n H_n(k) exp(-k^2/2) * (normalization)
    
    # The Fourier transform of H_n(x) exp(-x^2/2) is i^n H_n(k) exp(-k^2/2)
    # with the convention F(f)(k) = 1/sqrt(2*pi) integral f(x) exp(-ikx) dx
    
    # So F(g)(k) = 1/sqrt(2*pi) * sum c_n i^n H_n(k) exp(-k^2/2)
    
    # For even n, i^n = (-1)^(n/2)
    # H0: i^0 = 1
    # H4: i^4 = 1
    # H8: i^8 = 1
    # H12: i^12 = 1
    
    # So F(g)(k) = 1/sqrt(2*pi) * (c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)) * exp(-k^2/2)
    
    c12 = -(c0 * H0(0) + c4 * H4(0) + c8 * H8(0)) / H12(0)
    
    def g_func(x_val):
        poly = c0 * H0(x_val) + c4 * H4(x_val) + c8 * H8(x_val) + c12 * H12(x_val)
        return poly * np.exp(-x_val**2 / 2.0)
    
    def g_hat_func(k_val):
        poly = c0 * H0(k_val) + c4 * H4(k_val) + c8 * H8(k_val) + c12 * H12(k_val)
        return (1.0 / np.sqrt(2 * np.pi)) * poly * np.exp(-k_val**2 / 2.0)
    
    # Integrate |g(x)| dx
    A_g = np.trapz(np.abs(g_func(x)), x)
    
    # Integrate |F(g)(k)| dk
    A_g_hat = np.trapz(np.abs(g_hat_func(k)), k)
    
    return A_g * A_g_hat

def objective(coeffs):
    """Objective function to minimize."""
    c0, c4, c8 = coeffs
    try:
        bound = compute_bound(c0, c4, c8, n_grid=5000, x_max=12.0)
        if np.isnan(bound) or np.isinf(bound):
            return 1e10
        return bound
    except Exception:
        return 1e10

def construct():
    """Find optimal coefficients c0, c4, c8 to minimize the uncertainty bound."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial guess from the seed program
    x0 = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try multiple starting points
    start_points = [
        x0,
        np.array([-1.0, 0.01, 0.0001]),
        np.array([-0.5, 0.02, 0.001]),
        np.array([-2.0, 0.005, 0.0005]),
        np.array([1.0, -0.01, -0.0001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.2, 0.04, 0.0003]),
        np.array([-0.8, 0.03, 0.0008]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30.0:
            break
        
        try:
            res = minimize(objective, sp, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
    
    # Check time and do a final polish if time permits
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Ensure we return a valid result
    if time.time() > deadline - 5.0:
        # Return the best we have
        pass
    
    return best_x
# EVOLVE-BLOCK-END
