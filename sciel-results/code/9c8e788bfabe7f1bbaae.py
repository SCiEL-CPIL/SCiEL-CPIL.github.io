"""
Tune Hermite-polynomial coefficients to minimize the uncertainty bound C4.
We search over (a0, a4, a8) with a12 determined by g(0)=0, and minimize
the upper bound on A(f)*A(Fhat(f)) which corresponds to the L1/L2 norm ratio.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from math import factorial, pi

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 200.0  # keep margin before 210s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _hermite_phys(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(1, n):
        h2 = 2 * x * h1 - 2 * k * h0
        h0, h1 = h1, h2
    return h1


def _c4_bound(a0, a4, a8):
    """
    Compute the upper bound C4 for the test function
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    C4 = ||g||_1^2 / ||g||_2^2
    """
    # Physicist's Hermite polynomials at x=0:
    # H0(0)=1, H4(0)=12, H8(0)=10080, H12(0)=103219200
    a12 = -(a0 + 12.0 * a4 + 10080.0 * a8) / 103219200.0
    
    # Use a fine grid for accurate integration
    x = np.linspace(-15, 15, 30001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    h0 = _hermite_phys(0, x)
    h4 = _hermite_phys(4, x)
    h8 = _hermite_phys(8, x)
    h12 = _hermite_phys(12, x)
    
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    g = poly * np.exp(-x**2 / 2.0)
    
    # Compute L1 and L2 norms
    abs_g = np.abs(g)
    l1_norm = np.trapz(abs_g, x)
    l2_norm_sq = np.trapz(g**2, x)
    
    if l2_norm_sq < 1e-30:
        return 1e10
    
    # C4 = ||g||_1^2 / ||g||_2^2
    C4 = (l1_norm ** 2) / l2_norm_sq
    
    return C4


def _c4_bound_fast(a0, a4, a8):
    """
    Faster computation using analytical/semi-analytical approach.
    For g(x) = P(x) * exp(-x^2/2), compute norms via quadrature on polynomial.
    """
    a12 = -(a0 + 12.0 * a4 + 10080.0 * a8) / 103219200.0
    
    # P(x) coefficients (physicist's Hermite)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4608x^6 + 30240x^4 - 60480x^2 + 10080
    # H12 = 4096x^12 - 184320x^10 + 3932160x^8 - 42467520x^6 + 254016000x^4 - 765427200x^2 + 103219200
    
    p0 = 0.0  # by construction g(0)=0
    p2 = -48.0 * a4 - 60480.0 * a8 - 765427200.0 * a12
    p4 = 16.0 * a4 + 30240.0 * a8 + 254016000.0 * a12
    p6 = -4608.0 * a8 - 42467520.0 * a12
    p8 = 256.0 * a8 + 3932160.0 * a12
    p10 = -184320.0 * a12
    p12 = 4096.0 * a12
    
    p_coeffs = [p0, p2, p4, p6, p8, p10, p12]
    
    # For L2 norm: ||g||_2^2 = integral P(x)^2 * exp(-x^2) dx
    # Use Gauss-Hermite quadrature (exact for polynomials up to degree 2N-1)
    # P(x)^2 has degree 24, so we need N >= 13 for exactness
    # Use 20 points for safety
    
    # Gauss-Hermite nodes and weights for weight function exp(-x^2)
    # np.polynomial.hermite.hermgauss gives nodes/weights for exp(-x^2)
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(30)
    
    # Evaluate P at nodes
    P_vals = np.zeros_like(x_nodes)
    for k, pk in enumerate(p_coeffs):
        P_vals += pk * (x_nodes ** (2 * k))
    
    g_vals = P_vals * np.exp(-x_nodes**2 / 2.0)
    
    l2_norm_sq = np.sum(w_nodes * g_vals**2)
    
    if l2_norm_sq < 1e-30:
        return 1e10
    
    # For L1 norm: ||g||_1 = integral |P(x)| * exp(-x^2/2) dx
    # This is harder analytically due to the absolute value.
    # Use adaptive quadrature or a fine grid.
    
    # Use a fine grid for L1 (the absolute value makes it non-polynomial)
    x_grid = np.linspace(-12, 12, 20001)
    dx = x_grid[1] - x[0]
    
    P_grid = np.zeros_like(x_grid)
    for k, pk in enumerate(p_coeffs):
        P_grid += pk * (x_grid ** (2 * k))
    
    g_grid = P_grid * np.exp(-x_grid**2 / 2.0)
    
    l1_norm = np.trapz(np.abs(g_grid), x_grid)
    
    C4 = (l1_norm ** 2) / l2_norm_sq
    
    return C4


def _c4_bound_gh(a0, a4, a8):
    """
    Use Gauss-Hermite for L2 and adaptive Simpson for L1.
    """
    a12 = -(a0 + 12.0 * a4 + 10080.0 * a8) / 103219200.0
    
    p0 = 0.0
    p2 = -48.0 * a4 - 60480.0 * a8 - 765427200.0 * a12
    p4 = 16.0 * a4 + 30240.0 * a8 + 254016000.0 * a12
    p6 = -4608.0 * a8 - 42467520.0 * a12
    p8 = 256.0 * a8 + 3932160.0 * a12
    p10 = -184320.0 * a12
    p12 = 4096.0 * a12
    
    p_coeffs = [p0, p2, p4, p6, p8, p10, p12]
    
    # Gauss-Hermite for L2 (exact)
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(25)
    
    P_vals = np.zeros_like(x_nodes)
    for k, pk in enumerate(p_coeffs):
        P_vals += pk * (x_nodes ** (2 * k))
    
    g_vals = P_vals * np.exp(-x_nodes**2 / 2.0)
    l2_norm_sq = np.sum(w_nodes * g_vals**2)
    
    if l2_norm_sq < 1e-30:
        return 1e10
    
    # L1 via fine grid
    x_grid = np.linspace(-10, 10, 15001)
    P_grid = np.zeros_like(x_grid)
    for k, pk in enumerate(p_coeffs):
        P_grid += pk * (x_grid ** (2 * k))
    g_grid = P_grid * np.exp(-x_grid**2 / 2.0)
    l1_norm = np.trapz(np.abs(g_grid), x_grid)
    
    C4 = (l1_norm ** 2) / l2_norm_sq
    return C4


def construct():
    """
    Return the three coefficients (a0, a4, a8) that minimize C4.
    a12 is determined by g(0)=0.
    """
    # Seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    best_val = _c4_bound_gh(*x0)
    best_x = x0.copy()
    
    # Multi-start with diverse initial points
    rng = np.random.RandomState(42)
    
    starts = [x0]
    
    # Perturbations around seed
    for scale in [0.1, 0.3, 0.5, 1.0]:
        for _ in range(5):
            pert = rng.randn(3) * scale
            starts.append(x0 + pert)
    
    # Structured starts
    starts.append(np.array([-1.0, 0.05, 0.0]))
    starts.append(np.array([-1.2, 0.03, 0.001]))
    starts.append(np.array([-0.8, 0.08, 0.0005]))
    starts.append(np.array([-1.5, 0.02, 0.002]))
    
    for start in starts:
        if _time_left() < 15:
            break
        
        try:
            res = minimize(
                _c4_bound_gh,
                start,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Refine with L-BFGS-B (numerical gradient)
    if _time_left() > 10:
        try:
            res = minimize(
                _c4_bound_gh,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with very tight Nelder-Mead
    if _time_left() > 5:
        try:
            res = minimize(
                _c4_bound_gh,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
