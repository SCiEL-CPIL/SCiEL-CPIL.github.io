import numpy as np
from scipy.special import roots_genlaguerre
import time

_START_TIME = time.time()
_TIME_LIMIT = 550.0  # Safety margin before 600s limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

# Precompute factorials up to 24! for potential verification or alternative metrics
_MAX_DEG = 24
import math
_FACTS = np.array([math.factorial(k) for k in range(_MAX_DEG + 1)], dtype=float)

def _compute_l2_norm_sq(roots):
    """
    Compute the square of the L2 norm of P(x) = prod (x - r_i) with weight e^-x.
    This is kept for compatibility with the seed program's interface, 
    though the construct function will use the analytic optimum directly.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e100
    if np.any(roots <= 0):
        return 1e100
        
    try:
        # Coefficients of Q(x) = prod (x - r_i)
        coeffs_q = np.poly(roots)
        
        # Coefficients of P(x) = Q(x)^2
        coeffs_p = np.convolve(coeffs_q, coeffs_q)
        
        # result = sum b_k * k!
        result = np.dot(coeffs_p, _FACTS[::-1])
        
        return result
    except Exception:
        return 1e100

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    
    The optimal roots for minimizing the L2 norm of the monic polynomial 
    with respect to the Laguerre weight are the roots of the generalized 
    Laguerre polynomial L_12^(0)(x). We return these directly to avoid 
    numerical optimization errors and save time.
    """
    m = 12
    
    # The global minimum of int_0^inf (prod(x-r_i))^2 e^-x dx 
    # for monic polynomial of degree 12 is achieved by the Laguerre polynomial L_12^(0)(x).
    # Its roots are the optimal locations for r_i.
    
    try:
        # roots_genlaguerre(n, alpha) returns roots of L_n^(alpha)(x)
        # We need L_12^(0)(x)
        roots_opt = roots_genlaguerre(m, 0)
        
        # Ensure real parts (they should be real for alpha > -1)
        roots_opt = np.sort(roots_opt.real)
        
        # Verify validity
        if len(roots_opt) != 12:
            raise ValueError("Incorrect number of roots")
            
        if np.any(roots_opt <= 0):
            # Fallback if numerical issues arise, though unlikely for m=12, alpha=0
            raise ValueError("Roots must be positive")
            
        # Check distinctness
        if np.any(np.diff(roots_opt) < 1e-10):
            # They are distinct for Laguerre polynomials, but check just in case
            pass
            
        return [float(r) for r in roots_opt]
        
    except Exception:
        # Fallback to a heuristic distribution if scipy fails (highly unlikely)
        # This is a safety net, not the expected path
        return [float(4.0 * (i + 1)) for i in range(12)]
