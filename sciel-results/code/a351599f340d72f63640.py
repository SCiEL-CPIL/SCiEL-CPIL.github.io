"""Refined Cohn-Goncalves Laguerre construction: Optimized root search with gradients."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 600.0
_START = time.time()

def _time_left():
    return _DEADLINE - time.time()

def _laguerre_coeffs(n, alpha=1):
    """
    Compute coefficients of L_n^(alpha)(x) in ascending power of x.
    """
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = (math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1) - math.lgamma(k + alpha + 1))
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff

def _compute_norm_sq(poly):
    """
    Compute ||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx
    """
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    fact = np.ones(max_k + 2)
    for j in range(1, max_k + 2):
        fact[j] = fact[j - 1] * j
    weights = fact[1:max_k + 2]
    norm_sq = np.dot(sq, weights)
    return norm_sq

def _c4_bound(roots, m=12):
    """
    Compute upper bound on C4 for given roots.
    Returns the bound value (lower is better).
    """
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    
    return 1.0 / norm_sq

def _norm_and_grad(roots, m=12):
    """
    Compute norm_sq and its gradient with respect to roots.
    roots: array of 12 positive distinct values.
    """
    coeff_L = _laguerre_coeffs(m, 1)
    
    # Precompute factorials up to a high degree
    # Max degree of P is 1 + 2*12 = 25. P^2 is 50.
    # We need integrals of x^k P^2 e^-x.
    # In gradient calc, we need integrals of x^k R^2 e^-x where R is P/(x-r_i)^2.
    # Degree of R is 23. R^2 is 46.
    # We need moments up to k=2 for x^2 R^2.
    # So we need factorials up to 46+2 = 48.
    
    max_deg = 50
    fact = np.ones(max_deg + 10)
    for j in range(1, max_deg + 10):
        fact[j] = fact[j - 1] * j
    
    # Build P(x)
    P = coeff_L.copy()
    for r in roots:
        q = np.array([r * r, -2.0 * r, 1.0])
        P = np.convolve(P, q)
        
    P_sq = np.convolve(P, P)
    norm_sq = np.dot(P_sq, fact[1:len(P_sq)+1])
    
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 0.0, np.zeros(len(roots))
    
    grad = np.zeros(len(roots))
    
    for i, r_i in enumerate(roots):
        # Construct R(x) = P(x) / (x - r_i)^2
        # This is equivalent to building the product without the i-th term
        R = coeff_L.copy()
        for j, r_j in enumerate(roots):
            if j == i:
                continue
            q = np.array([r_j * r_j, -2.0 * r_j, 1.0])
            R = np.convolve(R, q)
            
        R_sq = np.convolve(R, R)
        
        # We need I = integral (x^2 - r_i x) R_sq(x) e^-x dx
        # = sum_k R_sq[k] ( (k+2)! - r_i (k+1)! )
        
        # (k+2)! is fact[k+2]
        # (k+1)! is fact[k+1]
        
        moments = R_sq * (fact[2:len(R_sq)+2] - r_i * fact[1:len(R_sq)+1])
        I = np.sum(moments)
        
        # dN/dr_i = -4 * I
        grad[i] = -4.0 * I
        
    return norm_sq, grad

def _neg_log_norm_and_grad(roots, m=12):
    """
    Objective and gradient for minimization.
    """
    if _time_left() < 2:
        return 1e100, np.zeros(len(roots))
        
    if np.any(roots <= 0):
        return 1e100, np.zeros(len(roots))
        
    norm_sq, grad_N = _norm_and_grad(roots, m)
    
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 1e100, np.zeros(len(roots))
        
    obj = -math.log(norm_sq)
    grad_obj = -grad_N / norm_sq
    
    # Add a small penalty for roots being too close to each other to ensure distinctness
    # and positivity (though bounds handle positivity)
    # Penalty: sum 1/(r_i - r_j)^2 for i<j
    # This is expensive to compute gradient, but 12 roots is small.
    # Actually, L-BFGS-B with bounds [1e-3, 100] should keep them away from 0.
    # Distinctness: if roots merge, the gradient might blow up or become unstable.
    # We'll rely on the optimization.
    
    return obj, grad_obj

def _make_roots(scale, alpha, n_roots=12):
    """Generate roots as scale * (i+1)^alpha for i=0..n_roots-1"""
    return [scale * ((i + 1) ** alpha) for i in range(n_roots)]

def construct():
    m = 12
    n_roots = 12
    
    # Baseline: the seed roots
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = 1.0
        
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Coarse grid over scale and exponent to find good starting points
    # We will store the top few best starting points for Phase 2
    candidates = []
    
    scales = np.linspace(1.0, 50.0, 30)
    alphas = np.linspace(0.3, 1.3, 30)
    
    for s in scales:
        if _time_left() < 300:
            break
        for a in alphas:
            if _time_left() < 300:
                break
            roots = _make_roots(s, a)
            if min(roots) <= 0:
                continue
            # Check distinctness
            if len(set(round(r, 9) for r in roots)) != 12:
                continue
            val = _c4_bound(roots, m)
            if np.isfinite(val):
                candidates.append((val, roots[:]))
    
    # Sort candidates by value
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local refinement using L-BFGS-B with gradient
    # We will optimize the top 5 candidates from the grid
    num_restarts = 5
    if len(candidates) < num_restarts:
        num_restarts = len(candidates)
    
    # Also add the seed as a candidate
    candidates.append((seed_val, seed_roots[:]))
    candidates.sort(key=lambda x: x[0])
    
    for val, roots in candidates[:num_restarts]:
        if _time_left() < 120:
            break
            
        x0 = np.array(roots)
        
        # Define the objective function for L-BFGS-B
        # We need to return (f, g)
        def fun_grad(x):
            # Ensure x is positive
            if np.any(x <= 0):
                return 1e100, np.zeros_like(x)
            return _neg_log_norm_and_grad(x, m)
            
        try:
            result = minimize(
                fun_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(1e-3, 100.0)] * n_roots,
                options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            
            if np.isfinite(result.fun):
                candidate_roots = result.x
                if all(r > 0 for r in candidate_roots) and len(set(round(r, 9) for r in candidate_roots)) == 12:
                    cand_val = _c4_bound(candidate_roots, m)
                    if np.isfinite(cand_val) and cand_val < best_val:
                        best_val = cand_val
                        best_roots = candidate_roots.tolist()
                        
        except Exception:
            pass

    # Phase 3: Final polish with even tighter tolerance
    if _time_left() > 60:
        x0 = np.array(best_roots)
        
        def fun_grad(x):
            if np.any(x <= 0):
                return 1e100, np.zeros_like(x)
            return _neg_log_norm_and_grad(x, m)
            
        try:
            result = minimize(
                fun_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(1e-3, 100.0)] * n_roots,
                options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-12}
            )
            
            if np.isfinite(result.fun):
                candidate_roots = result.x
                if all(r > 0 for r in candidate_roots) and len(set(round(r, 9) for r in candidate_roots)) == 12:
                    cand_val = _c4_bound(candidate_roots, m)
                    if np.isfinite(cand_val) and cand_val < best_val:
                        best_val = cand_val
                        best_roots = candidate_roots.tolist()
                        
        except Exception:
            pass

    # Ensure we return valid positive distinct roots
    if len(best_roots) != n_roots or any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
        
    return best_roots
# EVOLVE-BLOCK-END
