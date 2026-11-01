"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 180.0  # safety margin before 210s kill

def _hermite_physicist(n, x):
    """Physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    H_nm2 = np.ones_like(x)
    H_nm1 = 2.0 * x
    for k in range(2, n + 1):
        H_n = 2.0 * x * H_nm1 - 2.0 * (k - 1) * H_nm2
        H_nm2 = H_nm1
        H_nm1 = H_n
    return H_nm1

def _compute_objective(coeffs, x_grid):
    """Compute the upper bound on C4 = A(f) * A(F(f)).
    
    For even Hermite functions, F(f) = f (up to convention), so C4 ~ A(f)^2.
    We compute A(f) as the L1 norm of f(x) = sum a_i H_i(x) * exp(-x^2/2).
    """
    a0, a4, a8 = coeffs
    # a12 determined by g(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 4790016.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Compute f(x) on the grid
    H0 = _hermite_physicist(0, x_grid)
    H4 = _hermite_physicist(4, x_grid)
    H8 = _hermite_physicist(8, x_grid)
    H12 = _hermite_physicist(12, x_grid)
    
    f = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x_grid**2 / 2.0)
    
    # A(f) = integral of |f(x)| dx (L1 norm)
    A_f = np.trapz(np.abs(f), x_grid)
    
    # Since F(f) = f for even Hermite (standard convention), C4 = A_f * A_f
    C4 = A_f * A_f
    return C4

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Not enough time, return seed
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Set up grid for numerical integration
    x_grid = np.linspace(-20, 20, 10000)
    
    # Define objective for scipy minimize
    def objective(coeffs):
        return _compute_objective(coeffs, x_grid)
    
    # Seed initial guess
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_coeffs = x0.copy()
    best_val = objective(x0)
    
    # Try multiple starting points
    rng = np.random.RandomState(42)
    n_starts = 10
    for i in range(n_starts):
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE - 10.0:
            break
        
        if i == 0:
            start = x0.copy()
        else:
            start = x0 + rng.randn(3) * 0.1
        
        try:
            res = minimize(objective, start, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - _start_time
    if elapsed < _DEADLINE - 20.0:
        # Polish with tighter tolerance
        try:
            res = minimize(objective, best_coeffs, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
