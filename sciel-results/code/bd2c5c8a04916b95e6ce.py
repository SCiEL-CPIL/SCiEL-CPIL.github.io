"""
Optimized approach using Chebyshev basis expansion and L-BFGS-B.
The objective is value = abs(2 * N * max(convolve(h, h)) / sum(h)^2).
We parameterize h as a linear combination of Chebyshev polynomials to enforce smoothness
and reduce dimensionality, then optimize the coefficients.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_chebyt

N = 400
DEADLINE = 120.0
START_TIME = time.time()

# EVOLVE-BLOCK-START

def _value(h):
    s = np.sum(h)
    if np.isclose(s, 0):
        return 1e10
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _soft_max(x, beta=100.0):
    # Smooth approximation of max(x) using log-sum-exp
    # scaled by beta to control smoothness
    m = np.max(x)
    return m + np.log(np.sum(np.exp((x - m) * beta)) / len(x)) / beta

def _objective_soft(coeffs):
    # coeffs[0] is offset, coeffs[1:] are Chebyshev coefficients
    offset = coeffs[0]
    cheb_coeffs = coeffs[1:]
    
    # Generate Chebyshev basis
    # x in [-1, 1] corresponding to [-1/4, 1/4]
    x = np.linspace(-1, 1, N)
    
    h = offset * np.ones(N)
    for k, c in enumerate(cheb_coeffs, 1):
        h += c * eval_chebyt(k, x)
        
    # Ensure sum is not zero by adding a small regularization if needed
    s = np.sum(h)
    if np.isclose(s, 0):
        return 1e10
        
    c = np.convolve(h, h)
    # Use soft max for differentiability
    m = _soft_max(c, beta=50.0)
    return abs(2.0 * N * m / (s * s))

def _make_h_from_coeffs(coeffs):
    offset = coeffs[0]
    cheb_coeffs = coeffs[1:]
    x = np.linspace(-1, 1, N)
    h = offset * np.ones(N)
    for k, c in enumerate(cheb_coeffs, 1):
        h += c * eval_chebyt(k, x)
    return h

def construct():
    if time.time() - START_TIME > DEADLINE - 5:
        return np.ones(N)
        
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Number of Chebyshev terms to include
    num_terms = 5
    
    # Initial coefficients: mostly offset, small perturbations
    initial_coeffs = np.zeros(num_terms + 1)
    initial_coeffs[0] = 1.0
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    
    for _ in range(3):
        # Random small perturbations
        pert = rng.normal(0, 0.1, num_terms)
        coeffs0 = np.concatenate(([1.0], pert))
        
        res = minimize(_objective_soft, coeffs0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < best_val:
            h_candidate = _make_h_from_coeffs(res.x)
            val_candidate = _value(h_candidate)
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate
                
        # Check time
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
    return best_h

# EVOLVE-BLOCK-END
