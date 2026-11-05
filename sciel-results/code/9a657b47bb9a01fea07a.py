"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 10.0  # return 10s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m+1, m+1))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(m+1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    # Note: A is upper triangular. We solve A^T c = p.
    # Since we need gradients, we precompute the inverse or use solve.
    # For m=12, explicit inverse is fine and fast for repeated use in gradient calc.
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Weights for the norm: w_k = k+1
    weights = np.arange(m+1) + 1

    def compute_c_and_grad(log_roots):
        """
        Compute Laguerre coefficients c and their gradients wrt log_roots.
        Returns:
            c: array of shape (m+1,)
            dc_dlogr: array of shape (m+1, m) where dc_dlogr[k, i] = dc_k / dlog_r_i
        """
        roots = np.exp(log_roots)
        
        # Compute polynomial coefficients in standard basis (ascending)
        # P(x) = prod (x - r_i)
        # np.poly returns descending coeffs.
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending coeffs of P(x)
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Compute gradients dc/dlog_r_i
        # dc/dlog_r_i = A_T_inv @ (dp/dlog_r_i)
        # dp/dlog_r_i = dp/dr_i * dr_i/dlog_r_i = dp/dr_i * r_i
        # dp/dr_i = d/d r_i prod_{j=1}^m (x - r_j) = prod_{j \neq i} (x - r_j)
        
        dc_dlogr = np.zeros((m+1, m))
        
        for i in range(m):
            # Compute Q_i(x) = prod_{j \neq i} (x - r_j)
            # We can do this by dividing P(x) by (x - r_i) or constructing from other roots.
            # Constructing from other roots is safer numerically for small m.
            other_roots = np.delete(roots, i)
            q_coeffs_desc = np.poly(other_roots)
            q = q_coeffs_desc[::-1] # ascending
            
            # dp/dr_i = q
            # dp/dlog_r_i = r_i * q
            dp_dlogr_i = roots[i] * q
            
            # dc/dlog_r_i = A_T_inv @ dp_dlogr_i
            dc_dlogr[:, i] = A_T_inv @ dp_dlogr_i
            
        return c, dc_dlogr

    def objective_and_grad(log_roots):
        """
        Objective: Minimize 1 / ||P||^2.
        Equivalent to maximizing ||P||^2.
        We minimize -log(||P||^2) for numerical stability, or just -||P||^2.
        Let's minimize -||P||^2 directly as it is smooth.
        """
        c, dc_dlogr = compute_c_and_grad(log_roots)
        
        # Norm squared
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros(m)
        
        # We want to MINIMIZE the bound, which is proportional to 1/norm_sq.
        # So we minimize f = 1/norm_sq.
        # Gradient of f = -1/norm_sq^2 * d(norm_sq)/dlogr
        # d(norm_sq)/dlogr_i = 2 * sum_k c_k * w_k * dc_k/dlogr_i
        
        val = 1.0 / norm_sq
        
        # Compute gradient of val
        # d(val)/dlogr_i = - (1/norm_sq^2) * 2 * sum_k c_k w_k (dc_k/dlogr_i)
        
        # Vectorized computation of d(norm_sq)/dlogr
        # term_k = 2 * c_k * w_k
        terms = 2.0 * c * weights
        # d_norm_sq_dlogr[i] = sum_k terms_k * dc_dlogr[k, i]
        # This is terms @ dc_dlogr
        d_norm_sq_dlogr = terms @ dc_dlogr
        
        grad_val = - (1.0 / (norm_sq * norm_sq)) * d_norm_sq_dlogr
        
        return val, grad_val

    # Optimization Strategy:
    # Multi-start L-BFGS-B in log-space.
    
    best_roots = None
    best_val = 1e10
    
    # Generate diverse starting points
    # 1. Geometric sequence
    # 2. Clustered around 1
    # 3. Random log-uniform
    
    initializations = []
    
    # Geometric: r_i = A * q^i
    # Try different A and q
    for A0 in [0.5, 1.0, 2.0, 5.0]:
        for q in [1.2, 1.5, 2.0, 0.8]:
            roots = A0 * (q ** np.arange(m))
            if np.all(roots > 0):
                initializations.append(np.log(roots))
                
    # Clustered: r_i ~ N(mu, sigma)
    for mu in [0, 1, 2]:
        for sigma in [0.5, 1.0, 2.0]:
            rng = np.random.default_rng(42 + mu*10 + int(sigma*10))
            roots = rng.normal(mu, sigma, m)
            roots = np.abs(roots) + 0.1 # ensure positive
            initializations.append(np.log(roots))
            
    # Random
    rng = np.random.default_rng(123)
    for _ in range(10):
        roots = rng.uniform(0.1, 100.0, m)
        initializations.append(np.log(roots))

    # Shuffle initializations to mix types
    np.random.shuffle(initializations)
    
    for log_roots_init in initializations:
        if not time_ok():
            break
            
        try:
            # L-BFGS-B with analytical gradient
            res = minimize(
                objective_and_grad, 
                log_roots_init, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            continue
            
    # Fallback if all fail
    if best_roots is None:
        best_roots = 4.0 * (2.0 ** np.arange(m))
        
    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
