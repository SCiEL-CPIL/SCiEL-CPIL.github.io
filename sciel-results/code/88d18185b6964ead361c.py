"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import factorial, eval_laguerre
from scipy.optimize import minimize, brentq

# EVOLVE-BLOCK-START
_START_TIME = time.time()
DEADLINE = _START_TIME + 590.0  # safety margin before 630s kill

def _time_left():
    return DEADLINE - time.time()

def _laguerre_poly_coeffs(m, n):
    """Return coefficients of the generalized Laguerre polynomial L_m^{(n)}(x).
    Coeffs are in descending power order (numpy convention)."""
    coeffs = np.zeros(m + 1)
    # L_m^{(n)}(x) = sum_{k=0}^m (-1)^k * C(m+n, m-k) / k! * x^k
    # In descending power: coefficient of x^{m-k} is (-1)^k * C(m+n, m-k) / k!
    for k in range(m + 1):
        c = (-1.0) ** k * factorial(m + n, exact=False) / (factorial(m - k, exact=False) * factorial(k, exact=False))
        coeffs[m - k] = c
    return coeffs

def _construct_laguerre_combination(roots, m=12, n=1):
    """
    Construct the Laguerre combination P(x) = sum_i c_i * L_{m_i}^{(n)}(x)
    such that P has double roots at the given locations.
    
    For m=12, n=1: We need a polynomial of degree 12 with 12 double roots? 
    Wait, 12 double roots would require degree 24. But m=12 means we use L_12.
    
    Re-reading: "12 double roots with m=12". This likely means the combination
    is a linear combination of Laguerre polynomials up to order 12, and it has
    12 roots (each of multiplicity 2? No, that's 24). 
    
    Actually, for the Cohn-Goncalves construction, we typically look at a single
    Laguerre polynomial L_m^{(n)} which has m simple roots. The "double roots"
    might refer to the squared polynomial or a specific constraint.
    
    Let's assume the task implies finding 12 positive numbers r_i such that
    a specific Laguerre-related function has these as double roots.
    
    Given the seed returns [4, 8, ..., 48], let's assume we just need to return
    12 positive numbers. The score is the upper bound on C4.
    
    Since I don't have the exact scoring function, I will try to optimize the
    positions to minimize some proxy for the uncertainty bound. A common proxy
    is minimizing the maximum root or ensuring roots are well-distributed.
    
    However, the directive says "Keep this design". The seed design is a simple
    arithmetic progression. Let's try to refine it by optimizing the roots to
    minimize the "energy" or some related metric.
    
    Actually, looking at the seed: `return [4.0 * (i + 1) for i in range(12)]`.
    This is [4, 8, 12, ..., 48].
    
    Let's try to find roots that minimize the sum of squares or something similar,
    subject to being positive and distinct.
    
    Wait, the task says "verifies the root constraints". This implies there are
    constraints. Without knowing them, I'll assume the roots must be positive and
    perhaps related to the zeros of some Laguerre polynomial.
    
    Let's compute the actual zeros of L_12^{(1)}(x) and see if they are close to
    the seed values.
    """
    # Compute zeros of L_12^{(1)}(x)
    coeffs = _laguerre_poly_coeffs(12, 1)
    zeros = np.roots(coeffs)
    # Filter real positive zeros
    real_zeros = zeros[np.isreal(zeros)].real
    real_zeros = real_zeros[real_zeros > 0]
    real_zeros.sort()
    
    if len(real_zeros) == 12:
        return list(real_zeros)
    else:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]

def construct():
    """
    Return a list of 12 positive double-root locations.
    """
    # First, try the analytical zeros of L_12^{(1)}(x)
    try:
        coeffs = _laguerre_poly_coeffs(12, 1)
        zeros = np.roots(coeffs)
        real_zeros = zeros[np.isreal(zeros)].real
        real_zeros = real_zeros[real_zeros > 0]
        real_zeros.sort()
        
        if len(real_zeros) == 12:
            return list(real_zeros)
    except Exception:
        pass
    
    # Fallback: use the seed values
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
