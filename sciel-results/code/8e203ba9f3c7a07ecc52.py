<<<<<<< SEARCH
import time
import math
import numpy as np

def construct():
    """Return 12 double-root locations z_1..z_12."""
    # The task requires finding z_i such that B/(2*pi) is minimized.
    # B is the largest real root of q(x) = g(x) / (x * prod (x-z_i)^2) that is a simple root (sign change).
    # g(x) is a linear combination of L_0, L_2, ..., L_50 (generalized Laguerre with alpha=-1/2).
    # g satisfies g(0)=0, g'(0)=1, and g(z_i)=g'(z_i)=0.
    # This means g(x) has double roots at z_i and a simple root at 0.
    # Since deg(g) <= 50, and we have 12 double roots + 1 simple root = 25 roots.
    # The dimension of the space of such g is 26 (coefficients c_0..c_25).
    # The constraints are 1 + 1 + 12*2 = 26.
    # So for any distinct positive z_i, there is a unique g (up to scaling, but fixed by g'(0)=1).
    
    # The key insight is that we want to push the "next" root of g (which becomes a root of q) as close to 0 as possible?
    # No, we want to MINIMIZE B. B is the largest counted root of q.
    # q(x) is a polynomial of degree 50 - (1 + 24) = 25.
    # Wait, deg(g) is at most 50.
    # Roots of g: 0, z_1..z_12 (double). Total multiplicity 1 + 24 = 25.
    # So g(x) = x * prod (x-z_i)^2 * q(x).
    # deg(q) = 50 - 25 = 25.
    # q(x) is a polynomial of degree 25.
    # The roots of q are the "extra" roots of g.
    # We want to minimize the largest real root of q (that is simple).
    # Ideally, we want all roots of q to be negative or complex, so B=0.
    # Or at least, the largest real simple root to be as small as possible.
    
    # Let's try to find z_i such that q(x) has no positive real roots.
    # This is a hard optimization problem.
    
    # Heuristic:
    # The Laguerre polynomials L_n^{-1/2}(x) have positive roots.
    # The combination g is determined by the interpolation conditions.
    
    # Let's try a grid search or local optimization on z_i.
    # Given the time limit, we should be very careful.
    
    # Initial guess:
    # Maybe z_i are related to the roots of some specific polynomial?
    # If we choose z_i to be the roots of L_{12}^{-1/2}(x), does that help?
    # Or maybe we should just return a fixed set that works reasonably well.
    
    # Let's try to compute the score for a few candidate sets and pick the best.
    # But computing the score requires solving a 26x26 linear system and finding roots of a degree 25 polynomial.
    # This is fast in numpy.
    
    def _score(z_list):
=======
import time
import math
import numpy as np

# Precompute Laguerre coefficients at module level for efficiency
_LAGUERRE_COEFFS_CACHE = {}

def _get_laguerre_coeffs(n, alpha):
    """Get coefficients of L_n^alpha(x) using cache."""
    key = (n, alpha)
    if key in _LAGUERRE_COEFFS_CACHE:
        return _LAGUERRE_COEFFS_CACHE[key]
    
    if n == 0:
        result = np.array([1.0])
    elif n == 1:
        result = np.array([1.0 + alpha, -1.0])
    else:
        c_nm2 = np.array([1.0])
        c_nm1 = np.array([1.0 + alpha, -1.0])
        
        for k in range(2, n+1):
            t1 = c_nm1 * (2*k - 1 + alpha)
            t2 = np.zeros(len(c_nm1) + 1)
            t2[1:] = -c_nm1
            t3 = c_nm2 * (-(k - 1 + alpha))
            
            max_len = max(len(t1), len(t2), len(t3))
            t1 = np.pad(t1, (0, max_len - len(t1)))
            t2 = np.pad(t2, (0, max_len - len(t2)))
            t3 = np.pad(t3, (0, max_len - len(t3)))
            
            c_n = (t1 + t2 + t3) / k
            c_nm2 = c_nm1
            c_nm1 = c_n
        result = c_nm1
    
    _LAGUERRE_COEFFS_CACHE[key] = result
    return result

def construct():
    """Return 12 double-root locations z_1..z_12."""
    # Precompute all needed Laguerre coefficients
    alpha = -0.5
    L_coeffs = [_get_laguerre_coeffs(2*k, alpha) for k in range(26)]
    
    # Precompute derivative coefficient arrays
    L_deriv_coeffs = []
    for k in range(26):
        lc = L_coeffs[k]
        if len(lc) > 1:
            d_coeffs = np.array([j * lc[j] for j in range(1, len(lc))])
        else:
            d_coeffs = np.array([])
        L_deriv_coeffs.append(d_coeffs)
    
    # Precompute the matrix structure (A depends on z_i, but we can precompute parts)
    # Actually, A depends on z_i, so we need to build it each time.
    # But we can precompute the Laguerre polynomial evaluations more efficiently.
    
    def _score(z_list, L_coeffs=L_coeffs, L_deriv_coeffs=L_deriv_coeffs):
>>>>>>> REPLACE

