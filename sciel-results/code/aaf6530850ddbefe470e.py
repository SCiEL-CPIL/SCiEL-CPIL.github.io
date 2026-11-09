"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_genlaguerre, genlaguerre, eval_genlaguerre
from scipy.optimize import minimize
import warnings

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 uncertainty bound given 12 double-root locations.
    
    The construction involves a polynomial P(x) with double roots at the given locations.
    For the Fourier sign uncertainty, we consider the Laguerre combination.
    The C4 bound is related to the ratio of moments.
    
    We model P(x) = prod_i (x - r_i)^2 * L_m^{(n)}(x) or similar construction.
    Actually, for the Cohn-Goncalves construction, we tune the roots to minimize
    the uncertainty bound.
    
    The score is the upper bound on C4. Lower is better.
    
    Using Gauss-Laguerre quadrature to compute moments:
    - The weight function is e^{-x} for Laguerre polynomials.
    - We compute the relevant moments of the polynomial combination.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return 1e10
    
    # Ensure roots are positive and sorted
    if np.any(roots <= 0):
        return 1e10
    
    # For the Cohn-Goncalves construction with m=12, n=1:
    # We have a polynomial that is a combination of Laguerre polynomials.
    # The double roots are at the given locations.
    
    # The uncertainty bound C4 involves the ratio:
    # C4 = (integral of x^2 * |f|^2 dx) / (integral of |f|^2 dx) * (something)
    # Actually, for sign uncertainty, it's more complex.
    
    # Let's use the known structure:
    # The polynomial is P(x) = L_12^{(1)}(x) * prod_i (x - r_i)^2 / some normalization
    # But actually, the roots ARE the double roots of the Laguerre combination.
    
    # A simpler model: the C4 bound for a function with zeros at these locations
    # in the Laguerre space can be computed via the reproducing kernel or moments.
    
    # Using the fact that for the refined construction, the bound is:
    # C4 = sum of weights at roots / some reference
    
    # Let's compute using Gauss-Laguerre quadrature with enough points
    # to capture the polynomial structure.
    
    # Number of quadrature points - needs to be high enough for degree ~24+12=36
    n_quad = 64
    
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(n_quad)
    
    # The polynomial P(x) = prod_i (x - r_i)^2 * L_m^{(n)}(x)
    # But actually, for the uncertainty bound, we need:
    # The function f(x) = P(x) * e^{-x/2} (to be in L2 with weight e^{-x})
    
    # Let's compute the moments:
    # M_k = integral_0^inf x^k * P(x)^2 * e^{-x} dx
    
    # Using Gauss-Laguerre:
    # M_k = sum_i w_i * (x_i)^k * P(x_i)^2
    
    # Compute P(x_i) for all quadrature nodes
    # P(x) = prod_i (x - r_i)^2 * L_12^{(1)}(x)
    
    # Actually, let me reconsider. The task says "12 double roots with m=12 and dimension n=1".
    # This likely means we have a polynomial of degree 12 (m=12) with some parameter n=1,
    # and we're placing 12 double roots.
    
    # The C4 bound for the Fourier sign uncertainty in this context is:
    # C4 = (integral x^2 |f|^2) / (integral |f|^2) where f is the constructed function
    
    # Let's compute:
    # f(x) = L_12^{(1)}(x) * prod_i (x - r_i)^2 * e^{-x/2}
    
    # But that would be degree 12+24=36, which is fine.
    
    # Actually, I think the construction is:
    # We have a polynomial Q(x) of degree 12 that has double roots at r_1,...,r_12
    # But degree 12 with 12 double roots means degree 24, not 12.
    
    # Let me re-read: "12 double roots with m=12 and dimension n=1"
    # This might mean the Laguerre polynomial parameter is m=12, alpha=n=1,
    # and we're tuning 12 root locations.
    
    # For the purpose of this optimization, let's define the bound as:
    # C4 = (sum_i w_i * x_i^2 * P(x_i)^2) / (sum_i w_i * P(x_i)^2)
    # where P(x) = prod_j (x - r_j)^2 * L_12^{(1)}(x)
    
    # This gives us a concrete objective to minimize.
    
    # Compute P(x_i) for all nodes
    P_vals = np.ones(n_quad)
    for r in roots:
        P_vals *= (x_nodes - r) ** 2
    
    # Multiply by L_12^{(1)}(x)
    L_vals = eval_genlaguerre(12, 1, x_nodes)
    P_vals *= L_vals
    
    # Compute moments
    M0 = np.sum(x_weights * P_vals ** 2)
    M2 = np.sum(x_weights * x_nodes ** 2 * P_vals ** 2)
    
    if M0 < 1e-300:
        return 1e10
    
    C4 = M2 / M0
    return C4

def _objective(params):
    """Objective function for optimization."""
    roots = np.exp(params)  # Ensure positivity via exponential transform
    return _compute_c4_bound(roots)

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive values.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Initial guess: zeros of (L_12^{(1)}(x))^2
    x_zeros, _ = roots_genlaguerre(12, 1)
    # These are the simple zeros of L_12^{(1)}, we use them as initial double-root locations
    initial_roots = x_zeros.copy()
    
    # Ensure they're positive (they should be for Laguerre)
    initial_roots = np.maximum(initial_roots, 1e-10)
    
    # Convert to log-space for optimization
    log_params = np.log(initial_roots)
    
    best_params = log_params.copy()
    best_score = _objective(log_params)
    
    # Try multiple optimization methods and restarts
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            params = best_params.copy()
        else:
            # Perturb the best solution
            params = best_params + 0.1 * np.random.randn(12)
        
        for method in methods:
            if time.time() > deadline - 30:
                break
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        _objective,
                        params,
                        method=method,
                        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
                    
            except Exception:
                continue
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Convert back from log-space
    final_roots = np.exp(best_params)
    
    # Ensure roots are positive and sorted
    final_roots = np.maximum(final_roots, 1e-10)
    final_roots = np.sort(final_roots)
    
    return [float(r) for r in final_roots]
# EVOLVE-BLOCK-END
