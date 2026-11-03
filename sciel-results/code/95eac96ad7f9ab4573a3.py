"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 180.0  # seconds, with safety margin from 210s kill

def _hermite_powers(x, n):
    """Compute x^n efficiently for polynomial evaluation."""
    return x ** n

def _eval_poly(coeffs, x):
    """Evaluate polynomial with given coefficients (ascending power) at x."""
    result = np.zeros_like(x, dtype=np.float64)
    for c in reversed(coeffs):
        result = result * x + c
    return result

def _hermite_gaussian(x, c0, c4, c8, c12):
    """
    Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    Using explicit polynomial forms.
    """
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x8 * x4
    
    H0 = np.ones_like(x)
    H4 = x4 - 6.0 * x2 + 3.0
    H8 = x8 - 28.0 * x6 + 210.0 * x4 - 420.0 * x2 + 105.0
    H12 = x12 - 60.0 * x10 + 1260.0 * x8 - 12600.0 * x6 + 45360.0 * x4 - 55440.0 * x2 + 10395.0
    
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    return poly * np.exp(-0.5 * x2)

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function.
    The H12 coefficient is determined by the constraint g(0) = 0.
    
    For a function g(x) = P(x) * exp(-x^2/2), the Fourier transform 
    (with convention F[f](xi) = int f(x) exp(-i x xi) dx) is:
    F[g](xi) = (2*pi)^(1/2) * P(i*xi) * exp(-xi^2/2) ... 
    Actually, let me use the standard result: if g(x) = sum a_n H_n(x) exp(-x^2/2),
    then F[g](xi) = (2*pi)^(1/2) * sum a_n (i)^n H_n(xi) exp(-xi^2/2)
    
    So |F[g](xi)| = (2*pi)^(1/2) * |sum a_n (i)^n H_n(xi) exp(-xi^2/2)|
    
    The uncertainty product involves ||g||_inf * ||F[g]||_inf.
    """
    # Compute c12 from constraint g(0) = 0
    # g(0) = (c0*1 + c4*3 + c8*105 + c12*10395) * exp(0) = 0
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
    
    # Evaluate on a grid to find the maximum
    # Use a fine grid around the origin
    x_grid = np.linspace(-10, 10, 20001)
    
    # Compute g(x)
    g_vals = _hermite_gaussian(x_grid, c0, c4, c8, c12)
    g_max = np.max(np.abs(g_vals))
    
    # Compute Fourier transform analytically
    # F[g](xi) = sqrt(2*pi) * sum_n a_n (i)^n H_n(xi) exp(-xi^2/2)
    # For our polynomial:
    # a0 = c0, a4 = c4, a8 = c8, a12 = c12
    # (i)^0 = 1, (i)^4 = 1, (i)^8 = 1, (i)^12 = -1
    # So F[g](xi) = sqrt(2*pi) * (c0*H0(xi) + c4*H4(xi) + c8*H8(xi) - c12*H12(xi)) * exp(-xi^2/2)
    
    xi_grid = np.linspace(-10, 10, 20001)
    xi2 = xi_grid * xi_grid
    xi4 = xi2 * xi2
    xi6 = xi4 * xi2
    xi8 = xi4 * xi4
    xi10 = xi8 * xi2
    xi12 = xi8 * xi4
    
    H0_xi = np.ones_like(xi_grid)
    H4_xi = xi4 - 6.0 * xi2 + 3.0
    H8_xi = xi8 - 28.0 * xi6 + 210.0 * xi4 - 420.0 * xi2 + 105.0
    H12_xi = xi12 - 60.0 * xi10 + 1260.0 * xi8 - 12600.0 * xi6 + 45360.0 * xi4 - 55440.0 * xi2 + 10395.0
    
    # Note: (i)^12 = (i^4)^3 = 1^3 = 1... wait, i^12 = (i^4)^3 = 1
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = 1
    # So all signs are positive!
    ft_poly = c0 * H0_xi + c4 * H4_xi + c8 * H8_xi + c12 * H12_xi
    ft_vals = np.sqrt(2.0 * np.pi) * ft_poly * np.exp(-0.5 * xi2)
    ft_max = np.max(np.abs(ft_vals))
    
    # The product is g_max * ft_max
    # But we need to be careful about what "upper bound on C4" means
    # C4 is likely the constant in the uncertainty principle
    # The product ||g||_inf * ||F[g]||_inf gives a lower bound on the constant
    # We want to minimize this product to get the tightest bound
    
    return g_max * ft_max

def construct():
    """Return the three optimal coefficients [c0, c4, c8]."""
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 10.0:
        # Not enough time, return prior best
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Start from the prior best and try to improve
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_c = x0.copy()
    best_val = _compute_bound(*x0)
    
    def objective(c):
        return _compute_bound(c[0], c[1], c[2])
    
    # Try multiple starting points and optimization methods
    starts = [
        x0,
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 1.0, -0.1]),
        np.array([-0.5, 0.5, 0.0]),
        np.array([-2.0, 0.5, 0.1]),
        np.array([-1.0, -0.5, 0.2]),
    ]
    
    for start in starts:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - 30.0:
            break
        
        try:
            result = minimize(
                objective, start, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerances if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 20.0 and best_c is not None:
        try:
            result = minimize(
                objective, best_c, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_c = result.x
        except Exception:
            pass
    
    return np.array(best_c)

# EVOLVE-BLOCK-END
