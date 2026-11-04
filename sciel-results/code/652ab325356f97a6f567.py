"""Refined Cohn-Goncalves Laguerre construction: 12-dim log-space DE + L-BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    m = 12
    # We need coefficients up to degree m (which is 12)
    # The polynomial is of degree 12 (product of 12 linear terms)
    # L_k is degree k. So we need L_0 to L_12.
    # The matrix A maps Laguerre coeffs to standard coeffs.
    # We want c such that p = A @ c. So c = A^{-1} @ p.
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            # Coeff of x^j in L_k^(1)(x)
            # Formula: (-1)^j * binom(k+1, j+1) / j!
            try:
                A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
            except Exception:
                A[k, j] = 0.0
                
    # Solve for c given p. c = A^{-1} @ p
    # Note: A is lower triangular (if we view rows as k, cols as j, A[k,j] is non-zero for j<=k)
    # Actually, standard basis coeffs p_j are linear combos of c_k.
    # p_j = sum_{k=j}^m A[k,j] c_k.
    # This is an upper triangular system for c if we order properly, or just invert the matrix.
    # Let's just invert the square matrix A (size (m+1)x(m+1)).
    # Wait, A[k,j] is coeff of x^j in L_k.
    # p(x) = sum_k c_k L_k(x) = sum_k c_k sum_j A[k,j] x^j = sum_j (sum_k A[k,j] c_k) x^j.
    # So p_j = sum_k A[k,j] c_k.
    # In matrix form: p = A^T c ? No.
    # Let M be the matrix where M[j, k] = A[k, j].
    # Then p = M c.
    # So c = M^{-1} p.
    # M is the transpose of A.
    
    M = A.T
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # Ensure p is length m+1
        if len(p) < m + 1:
            p = np.pad(p, (0, m + 1 - len(p)))
        elif len(p) > m + 1:
            p = p[:m+1]
            
        # c = M_inv @ p
        c = M_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        weights = np.arange(m + 1) + 1
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
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

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Local refinement with L-BFGS-B.
    
    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i
    # log_r_i = log(4) + i * log(2)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    # Roots should be positive.
    # If roots are too small or too large, the norm might behave badly.
    # Based on seed (4 to 4*2^11 ~ 8192), log range is [~1.4, ~9].
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(10000) ~ 9.2
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    
    # Run Differential Evolution
    # We have ~570s. DE is stochastic.
    # Let's run with a reasonable popsize and maxiter.
    
    if _time_left() > 60:
        try:
            # Use smaller popsize and maxiter for speed, relying on polish
            # popsize=15, maxiter=100 is usually fast for 12D.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=15, 
                tol=1e-8, 
                polish=True,  # L-BFGS-B polish
                init='sobol', # Better initial distribution
                updating='deferred',
                workers=1
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
    
    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Phase 2: Local refinement with L-BFGS-B on the 12-parameter space
    if _time_left() > 30 and best_log_roots is not None:
        try:
            res = minimize(
                cost_func,
                x0=best_log_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
            
    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
