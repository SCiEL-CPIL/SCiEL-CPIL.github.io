"""Improved program: Optimize roots for Laguerre double-root construction."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre, gammaln
from scipy.optimize import minimize
from scipy.integrate import quad

M = 12          # number of double roots
N = 1           # Laguerre order (n=1)
K = 2 * M + 1   # number of basis Laguerres (degree up to 2M)

def _get_laguerre_coefficients(roots):
    """
    Computes the Laguerre coefficients c_k for P(x) = prod (x - r_i)^2
    expanded in the basis L_k^{(N)}(x).
    c_k = 1 / (k! Gamma(k+N+1)) * integral_0^inf x^N e^{-x} P(x) L_k^{(N)}(x) dx
    """
    # Precompute P(x) function
    def P(x):
        # P(x) = prod (x - r_i)^2
        # To avoid overflow/underflow issues in prod, we can compute log?
        # But x is in integration range, roots are ~10-50.
        # For x < min(roots), terms are positive.
        # We just multiply.
        val = 1.0
        for r in roots:
            term = (x - r) ** 2
            val *= term
        return val

    coeffs = np.zeros(K)
    for k in range(K):
        # Orthogonality constant for L_k^{(N)}
        # h_k = Gamma(k+N+1) / k!
        # c_k = 1/h_k * integral
        # log_h_k = gammaln(k+N+1) - gammaln(k+1)
        log_h_k = gammaln(k + N + 1) - gammaln(k + 1)
        
        # Integrand: x^N e^{-x} P(x) L_k^{(N)}(x)
        def integrand(x):
            # Handle x=0 if N=0, but N=1 so x^1 is fine.
            lag_val = eval_genlaguerre(k, N, x)
            return (x ** N) * np.exp(-x) * P(x) * lag_val

        # Integrate from 0 to inf
        # Use quad for accuracy
        try:
            integral, _ = quad(integrand, 0, np.inf, limit=100)
        except Exception:
            return None # Signal failure
        
        h_k = np.exp(log_h_k)
        coeffs[k] = integral / h_k
        
    return coeffs

def _validate_roots(roots):
    """Check if roots are valid and coefficients are non-negative."""
    if len(roots) != M:
        return False
    if np.any(roots <= 0):
        return False
    if np.any(np.diff(roots) <= 1e-6):
        return False
    
    coeffs = _get_laguerre_coefficients(roots)
    if coeffs is None:
        return False
    
    # Check non-negativity of coefficients
    # Allow small numerical tolerance
    if np.any(coeffs < -1e-8):
        return False
        
    return True

def construct():
    # Initial guess: roots of L_M^{(N)}(x)
    try:
        roots0, _ = roots_genlaguerre(M, N)
        roots0 = np.sort(roots0)
    except Exception:
        roots0 = np.array([i * 4.0 for i in range(1, M + 1)])

    # Objective: minimize the largest root
    # Variables: the M roots
    # Constraints: Laguerre coefficients of P(x) = prod (x-r_i)^2 must be non-negative
    
    def objective(params):
        roots = np.sort(params)
        # Penalty for invalid roots
        if np.any(roots <= 0):
            return 1e10
        if np.any(np.diff(roots) <= 1e-6):
            return 1e10
            
        coeffs = _get_laguerre_coefficients(roots)
        if coeffs is None:
            return 1e10
            
        # Penalty for negative coefficients
        neg_min = np.min(coeffs)
        if neg_min < 0:
            # Large penalty proportional to how negative it is
            return 1e6 * (-neg_min) + np.max(roots)
            
        return np.max(roots)

    # Use SLSQP or trust-constr
    # We don't have explicit inequality constraints in the standard form for SLSQP easily without defining them.
    # We can define constraints: coeffs >= 0.
    # But coeffs depend on roots non-linearly.
    # Let's use a penalty method with minimize.
    
    # Try multiple initial guesses to be safe
    best_roots = roots0
    best_score = objective(roots0)
    
    # Perturbations
    np.random.seed(42)
    for _ in range(5):
        perturbed = roots0 * (1 + 0.1 * np.random.randn(M))
        perturbed = np.sort(perturbed)
        if np.any(perturbed <= 0):
            continue
        score = objective(perturbed)
        if score < best_score:
            best_score = score
            best_roots = perturbed

    # Refine with minimize
    try:
        res = minimize(objective, best_roots, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_roots = res.x
            best_score = res.fun
    except Exception:
        pass

    # Final validation
    if not _validate_roots(best_roots):
        # Fallback to initial Laguerre roots if optimization broke validity
        if _validate_roots(roots0):
            best_roots = roots0
        else:
            # Ultimate fallback: uniform
            best_roots = np.array([i * 4.0 for i in range(1, M + 1)])

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
