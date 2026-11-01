"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Hermite polynomial coefficients (physicists' Hermite polynomials)
# We will use explicit evaluation functions for stability
def _H0(x):
    return np.ones_like(x)

def _H4(x):
    return x**4 - 6*x**2 + 3

def _H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def _H12(x):
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 56700*x**4 - 113400*x**2 + 10395

def _compute_poly_and_a3(a0, a1, a2):
    """
    Compute a3 such that P(0) = 0.
    P(0) = a0*H0(0) + a1*H4(0) + a2*H8(0) + a3*H12(0)
    H0(0) = 1
    H4(0) = 3
    H8(0) = 105
    H12(0) = 10395
    a0 + 3a1 + 105a2 + 10395a3 = 0
    """
    a3 = -(a0 * 1.0 + a1 * 3.0 + a2 * 105.0) / 10395.0
    return a3

def _objective(params):
    """
    Compute the upper bound on C4, which is max|g(x)| * max|F(g)(x)|.
    g(x) = (a0*H0(x) + a1*H4(x) + a2*H8(x) + a3*H12(x)) * exp(-x^2/2)
    F(g)(x) = (a0*H0(x) + a1*(-i)^4*H4(x) + a2*(-i)^8*H8(x) + a3*(-i)^12*H12(x)) * exp(-x^2/2)
    Note: (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
    Wait, let's check the Fourier transform convention.
    Standard physics convention: \hat{f}(k) = \int f(x) e^{-ikx} dx.
    \mathcal{F}[H_n(x) e^{-x^2/2}] = (-i)^n H_n(k) e^{-k^2/2} * \sqrt{2\pi}?
    Let's check normalization.
    If \hat{f}(k) = \frac{1}{\sqrt{2\pi}} \int f(x) e^{-ikx} dx, then \hat{e^{-x^2/2}} = e^{-k^2/2}.
    And \mathcal{F}[H_n(x) e^{-x^2/2}] = (-i)^n H_n(k) e^{-k^2/2}.
    
    So:
    g(x) = P(x) e^{-x^2/2}
    F(g)(k) = P_F(k) e^{-k^2/2}
    where P_F(k) = a0*H0(k) + a1*(1)*H4(k) + a2*(1)*H8(k) + a3*(1)*H12(k).
    
    Wait, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
    So P_F(k) is the same polynomial P(k)!
    
    Therefore, max|g(x)| = max|P(x) e^{-x^2/2}|
    and max|F(g)(k)| = max|P(k) e^{-k^2/2}|
    
    So the product is (max|P(x) e^{-x^2/2}|)^2.
    """
    a0, a1, a2 = params
    a3 = _compute_poly_and_a3(a0, a1, a2)
    
    # We need to maximize |P(x) e^{-x^2/2}|
    # P(x) = a0*H0(x) + a1*H4(x) + a2*H8(x) + a3*H12(x)
    
    # Use a grid to find the maximum
    # The function decays rapidly, so a range of [-10, 10] should be sufficient
    # But for safety, let's use [-15, 15]
    x = np.linspace(-15, 15, 10001)
    
    H0_vals = _H0(x)
    H4_vals = _H4(x)
    H8_vals = _H8(x)
    H12_vals = _H12(x)
    
    P_vals = a0 * H0_vals + a1 * H4_vals + a2 * H8_vals + a3 * H12_vals
    g_vals = P_vals * np.exp(-x**2 / 2.0)
    
    max_val = np.max(np.abs(g_vals))
    
    # The score is the upper bound, which is max_g * max_Fg = max_val^2
    return max_val**2

def construct():
    start_time = time.time()
    deadline = start_time + 170  # Safety margin
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    best_x = x0.copy()
    best_score = _objective(x0)
    
    # Use L-BFGS-B
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Try Nelder-Mead as a backup
    if time.time() < deadline - 30:
        try:
            nm_result = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_x = nm_result.x
        except Exception:
            pass
    
    # Try a few different starting points
    if time.time() < deadline - 60:
        for start in [
            np.array([0.0, 0.0, 0.0]),
            np.array([-0.5, 0.1, 0.001]),
            np.array([-2.0, 0.02, 0.0001]),
            np.array([-1.0, 0.05, 0.0005]),
        ]:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    _objective,
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x
            except Exception:
                pass
    
    return best_x
# EVOLVE-BLOCK-END
