"""
Refined uncertainty inequality: construct 12 positive double roots for the
Laguerre combination (m=12, n=1) to minimize the C4 upper bound.

This program combines:
  1. A structural change: using the polynomial P(x) = prod(x - r_i) (degree m=12)
     rather than the degree 2m=24 double-root polynomial, and using the
     generalized Laguerre basis L_k^{(1)} (alpha=1) as suggested by the inspirations.
     This aligns with the "n=1" parameter in the problem statement.
  2. The optimization machinery: full 12-dimensional optimization in log-space
     using Differential Evolution with multiple restarts and L-BFGS-B polish.
  3. Fast evaluation via precomputed transformation matrices.

The C4 bound is computed as 1 / ||P||^2 where P is expressed in the Laguerre basis
with the appropriate inner product for the weight x^1 * e^{-x}.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import roots_genlaguerre
import time
import math

_START_TIME = None


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _precompute_laguerre_transform(alpha=1):
    """
    Precompute the transformation matrix from monomial basis to Laguerre basis
    for L_k^{(alpha)}(x), k=0..m, and the inverse.
    
    L_k^{(alpha)}(x) = sum_{j=0}^k A[k,j] x^j
    where A[k,j] = (-1)^j * binom(k+alpha, k-j) / j!
    """
    m = 12
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            # A[k,j] = (-1)^j * binom(k+alpha, k-j) / j!
            # binom(k+alpha, k-j) = (k+alpha)! / ((k-j)! * (j+alpha)!)
            # For general alpha, use the product form:
            # binom(k+alpha, k-j) = prod_{i=0}^{k-j-1} (k+alpha-i) / (k-j)!
            # But more simply: binom(k+alpha, j+alpha) * j! / (k-j)! ... 
            # Let's use scipy's formula or direct computation.
            # Actually, the standard formula is:
            # L_k^{(alpha)}(x) = sum_{j=0}^k (-1)^j * binom(k+alpha, k-j) * x^j / j!
            # binom(k+alpha, k-j) for non-integer alpha:
            # = gamma(k+alpha+1) / (gamma(k-j+1) * gamma(j+alpha+1))
            # For integer alpha=1: binom(k+1, k-j) = binom(k+1, j+1)
            if alpha == 1:
                val = math.comb(k + 1, k - j)
            else:
                # General case using product
                val = 1.0
                for i in range(k - j):
                    val *= (k + alpha - i)
                val /= math.factorial(k - j)
                # This isn't quite right for non-integer alpha.
                # For alpha=1, let's just use the comb formula.
                # For other alpha values, we'll use a different approach.
                pass
            
            A[k, j] = ((-1) ** j) * val / math.factorial(j)
    
    # A is lower triangular: L_k(x) = sum_{j=0}^k A[k,j] x^j
    # We want to express p(x) = sum_{j=0}^m p_j x^j = sum_{k=0}^m c_k L_k^{(alpha)}(x)
    # In matrix form: p = A^T @ c, so c = (A^T)^{-1} @ p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    return A, A_T_inv


def _precompute_norm_weights(alpha=1):
    """
    Compute the norm weights for Laguerre polynomials L_k^{(alpha)}.
    ||L_k^{(alpha)}||^2 = integral_0^inf (L_k^{(alpha)}(x))^2 x^alpha e^{-x} dx
    = gamma(k+alpha+1) / k!
    For alpha=1: (k+1)! / k! = k+1
    """
    m = 12
    weights = np.zeros(m + 1)
    for k in range(m + 1):
        if alpha == 1:
            weights[k] = k + 1
        else:
            weights[k] = math.gamma(k + alpha + 1) / math.factorial(k)
    return weights


def construct():
    """
    Construct 12 positive double roots for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    The score is the upper bound on C4 (lower is better).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0
    
    def time_left():
        return DEADLINE - SAFETY_MARGIN - _elapsed()
    
    m = 12
    
    # Precompute transformation and norm weights for alpha=1
    A, A_T_inv = _precompute_laguerre_transform(alpha=1)
    norm_weights = _precompute_norm_weights(alpha=1)
    
    def compute_norm_sq(roots):
        """
        Compute the squared norm of P(x) = prod(x - r_i) in the Laguerre basis.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending: p[0] is constant term
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: sum c_k^2 * (k+1) for alpha=1
        norm_sq = np.sum(c**2 * norm_weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq
    
    def cost_func(log_roots):
        """
        Minimize the upper bound on C4 = 1 / ||P||^2.
        We return 1/norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
        
        return 1.0 / norm_sq
    
    # Seed: arithmetic sequence r_i = 4*(i+1) for i=0..11
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_log_roots = np.log(seed_roots)
    
    # Bounds for log_roots
    # Allow roots from 0.01 to 100000
    bounds = [(-4.6, 11.5)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None
    
    # Evaluate seed
    seed_val = cost_func(seed_log_roots)
    if seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots.copy()
        best_log_roots = seed_log_roots.copy()
    
    # Also try generalized Laguerre nodes as initial guesses
    # For various alpha values
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if time_left() < 10.0:
            break
        try:
            roots = roots_genlaguerre(12, alpha)
            roots = np.sort(roots)
            if np.any(roots <= 0):
                continue
            log_roots = np.log(roots)
            val = cost_func(log_roots)
            if val < best_val:
                best_val = val
                best_roots = roots.copy()
                best_log_roots = log_roots.copy()
        except Exception:
            continue
    
    # Run Differential Evolution with multiple restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if time_left() < 60.0:
            break
        
        try:
            result = differential_evolution(
                cost_func,
                bounds,
                seed=seed,
                maxiter=150,
                popsize=20,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass
    
    # Final high-precision polish pass
    if time_left() > 15.0 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func,
                best_log_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
        
        # Another round of DE with tighter bounds around the best solution
        if time_left() > 30.0 and best_log_roots is not None:
            try:
                # Tighten bounds around current best
                tight_bounds = []
                for i in range(12):
                    center = best_log_roots[i]
                    tight_bounds.append((center - 2.0, center + 2.0))
                
                result2 = differential_evolution(
                    cost_func,
                    tight_bounds,
                    seed=999,
                    maxiter=100,
                    popsize=15,
                    tol=1e-12,
                    polish=True,
                    init='latinhypercube'
                )
                
                if result2.fun < best_val:
                    best_val = result2.fun
                    best_log_roots = result2.x.copy()
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass
    
    # Ensure roots are positive and sorted
    if best_roots is None:
        best_roots = seed_roots
    
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
