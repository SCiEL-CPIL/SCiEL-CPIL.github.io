# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

def _compute_bound_analytic(coeffs):
    """
    Compute the upper bound on C4 using the analytic formula for Hermite-Gaussian functions.
    g(x) = sum c_n H_n(x) e^{-x^2/2}
    Var(x) = sum c_n^2 (n+0.5) w_n / sum c_n^2 w_n
    Var(k) = sum c_n^2 (2n+1) w_n / sum c_n^2 w_n
    where w_n = 2^n n!
    """
    a, b, c = coeffs
    
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0
    
    c_vals = np.array([a, b, c, d])
    n_vals = np.array([0, 4, 8, 12])
    
    # Weights w_n = 2^n n!
    # Use log to avoid overflow if necessary, but 12! is fine in float64
    # 2^12 * 12! ~ 2e12, which is fine.
    w_vals = (2.0 ** n_vals) * factorial(n_vals, exact=False)
    
    # v_n = n + 0.5
    v_vals = n_vals + 0.5
    
    # u_n = 2n + 1
    u_vals = 2.0 * n_vals + 1.0
    
    c_sq = c_vals * c_vals
    
    sum_w = np.sum(c_sq * w_vals)
    
    if sum_w < 1e-15:
        return 1e10
        
    sum_vw = np.sum(c_sq * v_vals * w_vals)
    sum_uw = np.sum(c_sq * u_vals * w_vals)
    
    var_x = sum_vw / sum_w
    var_k = sum_uw / sum_w
    
    return var_x * var_k

def construct():
    """Find optimal coefficients [a, b, c] minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Multi-start local optimization
    # The function is smooth and convex-like near the minimum.
    # We use several initial points to ensure global convergence.
    
    # Initial guesses
    # 1. Pure H0 (with d to satisfy g(0)=0)
    # 2. Balanced mix
    # 3. Random-ish
    
    initial_points = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.1, 0.1, 0.1],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    
    best_fun = 1e10
    best_x = np.array([1.0, 0.0, 0.0])
    
    for x0 in initial_points:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                _compute_bound_analytic,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x
        except Exception:
            continue
            
    # If L-BFGS-B failed or didn't improve, try Powell which is derivative-free
    # but robust for small dimensions
    if best_fun > 1e5:
        for x0 in initial_points:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    _compute_bound_analytic,
                    x0,
                    method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_x = res.x
            except Exception:
                continue

    return np.array(best_x)

# EVOLVE-BLOCK-END
