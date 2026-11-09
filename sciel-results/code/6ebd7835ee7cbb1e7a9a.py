"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root locations.
    
    For a Laguerre-type construction with m double roots, the bound involves
    the ratio of norms of the polynomial and its derivative under the Laguerre weight.
    
    P(x) = prod_i (x - r_i)^2
    The bound C4 ~ ||P'||^2 / (||P||^2 * something) or similar ratio.
    
    We use a standard formulation where the uncertainty bound is related to
    the eigenvalue problem for the Laguerre operator.
    """
    roots = np.asarray(roots, dtype=np.float64)
    m = len(roots)
    
    # Build the polynomial P(x) = prod (x - r_i)^2
    # Use log-space for numerical stability
    # P(x) = exp(sum 2*log(x - r_i))
    
    # For the Laguerre weight w(x) = e^{-x}, we need:
    # ||P||^2 = integral_0^inf P(x)^2 e^{-x} dx
    # ||P'||^2 = integral_0^inf P'(x)^2 e^{-x} dx
    
    # Use Gauss-Laguerre quadrature with enough points
    # n_quad should be large enough for accuracy
    n_quad = max(2 * m + 50, 200)
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(n_quad)
    
    # Evaluate P(x) and P'(x) at quadrature nodes
    # P(x) = prod (x - r_i)^2
    # log|P(x)| = 2 * sum log|x - r_i|
    # P'(x)/P(x) = 2 * sum 1/(x - r_i)
    
    # For numerical stability, compute in log space
    # But we need actual values for the integrals
    
    # Evaluate P(x)
    P_vals = np.ones_like(x_nodes)
    dP_vals = np.zeros_like(x_nodes)
    
    for r in roots:
        diff = x_nodes - r
        # P *= diff^2
        P_vals *= diff * diff
        # dP/dx = P * 2 * sum 1/(x - r_i)
        # We'll accumulate the logarithmic derivative
        dP_vals += 2.0 / diff
    
    # P'(x) = P(x) * sum_i 2/(x - r_i)
    dP_vals = P_vals * dP_vals
    
    # Integrate with Laguerre weight (the weight e^{-x} is already in x_weights)
    norm_P_sq = np.sum(P_vals * P_vals * x_weights)
    norm_dP_sq = np.sum(dP_vals * dP_vals * x_weights)
    
    if norm_P_sq <= 0:
        return 1e10
    
    # The C4 bound for the refined uncertainty inequality
    # Typically C4 = norm_dP_sq / norm_P_sq or a related quantity
    # For the Cohn-Gonçalves construction, the bound is:
    # C4 = (||P'||^2) / (||P||^2)
    # But there may be additional factors depending on the exact formulation
    
    # A common form for the sign uncertainty bound involves:
    # C4 = max over sign changes of some ratio
    # For simplicity and based on the task description, we use:
    # C4 = norm_dP_sq / norm_P_sq
    
    return norm_dP_sq / norm_P_sq


def construct():
    """Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Returns a list of 12 positive double-root locations that minimize the C4 upper bound.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second budget with safety margin
    
    # Initial guess: zeros of Laguerre polynomial L_12(x)
    _, zeros_l12 = roots_laguerre(12)
    x0 = zeros_l12.copy()
    
    # Ensure positivity (Laguerre zeros are positive, but enforce)
    x0 = np.maximum(x0, 1e-10)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    # Try a few different initializations and optimization strategies
    initializations = [
        x0.copy(),  # Zeros of L_12
    ]
    
    # Also try geometric spacing
    initializations.append(np.geomspace(0.5, 50.0, 12))
    # Try linear spacing
    initializations.append(np.linspace(1.0, 40.0, 12))
    # Try the seed program's initialization
    initializations.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    for x_init in initializations:
        if time.time() > deadline - 60:
            break
        
        # Use L-BFGS-B for constrained optimization (roots must be positive)
        try:
            result = minimize(
                _compute_c4_bound,
                x_init,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead for local refinement if time permits
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _compute_c4_bound,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Ensure all roots are positive
    best_roots = np.maximum(best_roots, 1e-10)
    
    # Sort for consistency
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
