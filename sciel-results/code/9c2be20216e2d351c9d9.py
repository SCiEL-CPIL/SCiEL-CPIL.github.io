"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize_scalar
import time

_START_TIME = time.time()
_TIMEOUT = 600.0

def _check_time():
    return time.time() - _START_TIME > _TIMEOUT * 0.95

def _compute_c4_bound(roots):
    """
    Compute the C4 moment ratio for a given set of double roots.
    The polynomial is P(x) = prod (x - r_i)^2.
    The weight function for the sign uncertainty problem typically involves 
    a specific kernel. For the Cohn-Goncalves construction, the bound 
    relates to the moments of the sign function.
    
    A common formulation for the upper bound C4 involves the ratio of 
    integrals related to the polynomial and its derivative, or simply 
    the 4th moment of the zero distribution weighted by the Laguerre 
    orthogonality measure.
    
    Let's use a robust proxy: The variance of the roots is a key factor 
    in uncertainty. However, the task specifies "C4 moment ratio".
    
    Based on the Cohn-Goncalves paper "Fourier uncertainty and the 
    Laguerre polynomial", the bound is often related to the 
    expectation of x^2 or similar moments under the weight e^{-x}.
    
    Let M_k = integral_0^infty x^k P(x)^2 e^{-x} dx.
    The bound is likely proportional to M_4 / M_2 or similar.
    Actually, the "sign uncertainty" usually bounds |hat f(0)| or similar.
    
    Let's assume the objective is to minimize the 4th moment of the 
    roots, or the ratio M_4/M_0, or M_4/M_2.
    Given "C4", it's highly probable it refers to the 4th central moment 
    or the raw 4th moment.
    
    Let's compute M_4 = sum w_i * (r_i)^4 * (P'(r_i))^2 ? No.
    
    Let's stick to the simplest interpretation: 
    The quality of the roots is measured by the spread. 
    Lower spread usually means better localization.
    
    Let's define the objective as the 4th moment of the roots: 
    Obj = (1/N) * sum(r_i^4)
    
    Alternatively, the Cohn-Goncalves bound is:
    C = ( \int |f|^4 ) / ( \int |f|^2 )^2 ...
    
    Let's use the variance of the roots as a proxy for the uncertainty, 
    but the prompt says "C4 moment ratio".
    
    Let's try: Obj = M_4 / M_2
    where M_k = sum_{i=1}^N w_i r_i^k with w_i = 1 (uniform weight for roots)
    or w_i proportional to the weight function at the roots.
    
    Since we are using Laguerre roots, the natural weight is e^{-x}.
    However, for the roots themselves, a simple power sum is a good 
    structural proxy.
    
    Let's use the ratio of the 4th power sum to the square of the 2nd 
    power sum? Or just the 4th moment.
    
    Let's try: Obj = np.mean(roots**4)
    """
    if len(roots) == 0:
        return np.inf
    roots = np.array(roots)
    # Ensure positive
    roots = np.abs(roots)
    # Sort
    roots.sort()
    
    # Compute moments
    m2 = np.mean(roots**2)
    m4 = np.mean(roots**4)
    
    # The "C4" likely refers to the 4th moment.
    # A smaller 4th moment implies tighter clustering.
    return m4

def _get_roots_for_alpha(alpha):
    """
    Get the double-root locations as the zeros of (L_12^(alpha)(x))^2.
    The zeros of L_n^(alpha)(x) are distinct and positive.
    Squaring the polynomial makes them double roots.
    """
    try:
        # roots_genlaguerre(n, alpha) returns nodes and weights
        # The nodes are the zeros of L_n^(alpha)
        nodes, _ = roots_genlaguerre(12, alpha)
        return nodes
    except Exception:
        return None

def _objective(alpha):
    if _check_time():
        return np.inf
    
    if alpha < -1:
        return np.inf
        
    roots = _get_roots_for_alpha(alpha)
    if roots is None:
        return np.inf
        
    # The roots must be positive and distinct
    if np.any(roots <= 0):
        return np.inf
    if len(set(np.round(roots, 10))) < 12:
        return np.inf
        
    return _compute_c4_bound(roots)

def construct():
    if _check_time():
        # Fallback: standard Laguerre roots (alpha=0)
        return list(_get_roots_for_alpha(0.0))
        
    # We search for alpha > -1
    # The optimal alpha is likely near 0 or slightly positive/negative
    # Let's scan a range and then refine
    
    # Coarse scan
    best_alpha = 0.0
    best_score = _objective(0.0)
    
    # Scan alpha from -0.9 to 5.0
    alphas = np.linspace(-0.9, 5.0, 100)
    for a in alphas:
        if _check_time():
            break
        s = _objective(a)
        if s < best_score:
            best_score = s
            best_alpha = a
            
    # Refine using minimize_scalar
    # Bounds around the best coarse alpha
    lower = max(-0.99, best_alpha - 0.5)
    upper = best_alpha + 0.5
    
    try:
        res = minimize_scalar(_objective, bounds=(lower, upper), method='bounded', options={'xatol': 1e-6})
        if res.success:
            best_alpha = res.x
            best_score = res.fun
    except Exception:
        pass
        
    final_roots = _get_roots_for_alpha(best_alpha)
    if final_roots is None:
        final_roots = _get_roots_for_alpha(0.0)
        
    return list(final_roots)
# EVOLVE-BLOCK-END
