"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre
import time

# EVOLVE-BLOCK-START
def _compute_last_sign_change(roots):
    """
    Compute the last sign change of the Cohn-Goncalves Laguerre construction.
    m=12 double roots, n=1 dimension.
    """
    m = len(roots)
    n = 1
    alpha = 2 * m - 1  # = 23
    
    # The Cohn-Goncalves construction:
    # F(x) = P(x) * [c0 * L_0^{(alpha)}(x) + c1 * L_1^{(alpha)}(x)]
    # where P(x) = prod_{i=1}^{m} (x - r_i)^2 >= 0
    # L_0^{(alpha)}(x) = 1, L_1^{(alpha)}(x) = -x + alpha + 1
    # F(x) = P(x) * (c0 + c1*(-x + alpha + 1)) = P(x) * (A - B*x)
    # A = c0 + c1*(alpha+1), B = c1
    # Sign change at x = A/B (if B != 0)
    
    # Coefficients determined by moment constraint:
    # integral_0^inf F(x) * e^{-x} dx = 0 (for n=1)
    # => c0 * I0 + c1 * I1 = 0
    # where I0 = int P(x)*e^{-x} dx, I1 = int P(x)*L_1^{(alpha)}(x)*e^{-x} dx
    
    # Using Gauss-Laguerre quadrature with weight x^alpha * e^{-x}:
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) at quadrature points
    P_vals = np.ones(len(x_quad))
    for r in roots:
        P_vals *= (x_quad - r) ** 2
    
    L1_vals = -x_quad + alpha + 1
    
    weight = (x_quad ** alpha) * np.exp(-x_quad)
    
    I0 = np.sum(P_vals * weight * w_quad)
    I1 = np.sum(P_vals * L1_vals * weight * w_quad)
    
    if abs(I0) < 1e-50:
        return max(roots)
    
    # c0/I0 + c1*I1/I0 = 0 => c0 = -c1 * I1/I0
    # A = c0 + c1*(alpha+1) = c1*(-I1/I0 + alpha+1)
    # B = c1
    # A/B = -I1/I0 + alpha + 1
    
    T = -I1 / I0 + alpha + 1
    
    return max(0.0, T)

def _objective(roots):
    """Objective: minimize the last sign change (C4 upper bound)."""
    try:
        val = _compute_last_sign_change(roots)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    """Return 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    m = 12
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    bounds = [(0.01, 300.0)] * m
    
    best_roots = x0.copy()
    best_score = _objective(best_roots)
    
    # Primary SLSQP optimization
    try:
        result = minimize(
            _objective, x0, method='SLSQP', bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-14}
        )
        if np.isfinite(result.fun) and result.fun < best_score:
            best_roots = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Multi-start with diverse initializations
    start_points = [
        np.linspace(1.0, 15.0, m),
        np.logspace(-0.3, np.log10(20.0), m),
        np.array([0.5, 1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56.0]),
        np.array([2.0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24.0]),
        np.linspace(0.5, 8.0, m),
        np.array([1.0, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21.0]),
        np.linspace(1.0, 25.0, m),
        np.array([0.8, 1.5, 2.5, 4.0, 6.0, 8.5, 11.5, 15.0, 19.0, 24.0, 30.0, 37.0]),
    ]
    
    for sp in start_points:
        if time.time() - start_time > deadline - 15:
            break
        try:
            res = minimize(
                _objective, sp, method='SLSQP', bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14}
            )
            if np.isfinite(res.fun) and res.fun < best_score:
                best_roots = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Polish with tighter tolerance if time permits
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(
                _objective, best_roots, method='SLSQP', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            if np.isfinite(res.fun) and res.fun < best_score:
                best_roots = res.x
                best_score = res.fun
        except Exception:
            pass
    
    best_roots = np.sort(np.maximum(best_roots, 0.01))
    return list(best_roots)
# EVOLVE-BLOCK-END
