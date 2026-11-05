import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 15.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m+1, m+1))
    for k in range(m+1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    # Note: A is upper triangular. We solve A^T c = p.
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Weights for the norm: w_k = k+1
    weights = np.arange(m+1) + 1

    def compute_norm_sq(log_roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        """
        roots = np.exp(log_roots)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        return norm_sq

    def objective(log_roots):
        """
        Objective to minimize: -log(norm_sq) or just -norm_sq.
        Using -norm_sq directly.
        """
        norm_sq = compute_norm_sq(log_roots)
        if norm_sq <= 1e-10:
            return 1e10
        return -norm_sq

    def objective_and_grad(log_roots):
        """
        For refinement with L-BFGS-B.
        """
        roots = np.exp(log_roots)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros(m)
        
        val = -norm_sq
        
        # Gradient calculation
        dc_dlogr = np.zeros((m+1, m))
        for i in range(m):
            other_roots = np.delete(roots, i)
            q_coeffs_desc = np.poly(other_roots)
            q = q_coeffs_desc[::-1]
            dp_dlogr_i = roots[i] * q
            dc_dlogr[:, i] = A_T_inv @ dp_dlogr_i
        
        terms = 2.0 * c * weights
        d_norm_sq_dlogr = terms @ dc_dlogr
        grad_val = -d_norm_sq_dlogr
        
        return val, grad_val

    # Optimization Strategy:
    # 1. Differential Evolution for global exploration
    # 2. L-BFGS-B for local refinement
    
    # Bounds for log-roots
    # Roots can range from 1e-3 to 1e3
    bounds = [(-7.0, 7.0) for _ in range(m)]
    
    # Initial population for DE can be random, but let's seed it with some structured points
    # We'll let DE handle the initialization mostly, but ensure it runs within time.
    
    best_val = 1e10
    best_log_roots = None
    
    # Use a wrapper to track time
    de_timeout = 400.0
    de_start = time.time()
    
    # Differential Evolution
    # maxiter is tricky with time limit. We'll run DE for a fixed number of iterations
    # and check time.
    # Population size 100, Maxiter 1000 might be too slow.
    # Let's estimate: 12 dims, pop 100. Each eval ~1ms? 1000 iter * 100 pop = 100k evals.
    # 100k * 1ms = 100s. So 1000 iterations is safe.
    
    try:
        # Use a callback to stop if time is up?
        # scipy DE doesn't have a direct time-based stop, but we can limit maxiter.
        # Let's run DE with maxiter=500.
        de_res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=500, 
            popsize=15, # 15 * 12 = 180 individuals
            tol=1e-6, 
            mutation=(0.5, 1.0),
            recombination=0.5,
            polish=False # We will polish manually
        )
        
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_log_roots = de_res.x
            
    except Exception:
        pass
    
    # Refine the best result from DE and a few other good ones if available
    # DE returns only the best. Let's also try refining some random good points if time permits.
    
    candidates = []
    if best_log_roots is not None:
        candidates.append(best_log_roots)
    
    # Add some structured initializations for refinement if time allows
    if time_ok():
        # Geometric
        for A0 in [0.5, 1.0, 2.0]:
            for q in [1.5, 2.0]:
                roots = A0 * (q ** np.arange(m))
                candidates.append(np.log(roots))
        
        # Random
        rng = np.random.default_rng(123)
        for _ in range(5):
            roots = rng.uniform(0.1, 100.0, m)
            candidates.append(np.log(roots))
            
    for cand in candidates:
        if not time_ok():
            break
        try:
            res = minimize(
                objective_and_grad, 
                cand, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
        except Exception:
            continue
            
    if best_log_roots is None:
        # Fallback
        best_log_roots = np.log(4.0 * (2.0 ** np.arange(m)))
        
    best_roots = np.exp(best_log_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
