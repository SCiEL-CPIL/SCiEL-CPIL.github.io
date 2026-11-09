import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1  # alpha = 1
    max_deg = 2 * m  # 24, since roots are double

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial P(x) = prod (x - r_i)^2 
        in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial Q(x) = prod(x - r_i)
        # np.poly returns descending coeffs
        coeffs_q_desc = np.poly(roots)
        p_q = coeffs_q_desc[::-1] # ascending
        
        # P(x) = Q(x)^2
        p_p = np.convolve(p_q, p_q)
        
        # Ensure length is max_deg + 1
        if len(p_p) < max_deg + 1:
            p_p = np.pad(p_p, (0, max_deg + 1 - len(p_p)))
        elif len(p_p) > max_deg + 1:
            p_p = p_p[:max_deg + 1]
            
        # c = A_T_inv @ p_p
        c = A_T_inv @ p_p
        
        # Norm squared: sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(max_deg + 1) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        """
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    best_val = 1e10
    best_roots = None

    # Initial candidates
    # Use roots of Laguerre L_12^(1) as a starting point
    try:
        from scipy.special import roots_laguerre
        base_roots = roots_laguerre(12, 1)[0]
        if np.all(base_roots > 0):
            best_roots = base_roots.copy()
            best_val = cost_func(best_roots)
    except Exception:
        pass

    if best_roots is None:
        # Fallback to geometric progression
        best_roots = np.array([1.0 * (2.0 ** i) for i in range(12)])
        best_val = cost_func(best_roots)

    # Optimize using L-BFGS-B on log(roots)
    if time.time() < deadline - 50:
        log_roots_init = np.log(best_roots)
        
        def cost_log_roots(lr):
            r = np.exp(lr)
            return cost_func(r)
            
        # Use multiple restarts to avoid local minima
        n_restarts = 5
        for i in range(n_restarts):
            if time.time() > deadline - 20:
                break
            if i == 0:
                current_log_roots = log_roots_init
            else:
                # Perturb initial guess
                current_log_roots = log_roots_init + np.random.normal(0, 0.5, 12)
            
            try:
                res = minimize(
                    cost_log_roots,
                    current_log_roots,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_roots = np.exp(res.x)
                    log_roots_init = res.x
            except Exception:
                pass

    if best_roots is None:
        best_roots = np.array([1.0] * 12)
        
    return list(best_roots)
