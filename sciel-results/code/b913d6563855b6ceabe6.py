import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1  # alpha = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_log_norm_sq(roots):
        """
        Compute the log of the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return -1e10
        
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return -1e10
        return math.log(norm_sq)

    def cost_func(roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        We minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        log_norm_sq = compute_log_norm_sq(roots)
        if log_norm_sq < -1e9:
            return 1e10
        return math.exp(-log_norm_sq)

    # Global Search using Differential Evolution
    # Bounds for roots: [1e-5, 1e5]
    bounds = [(1e-5, 1e5)] * 12
    
    # We minimize -log_norm_sq to maximize norm_sq
    def de_cost(roots):
        return -compute_log_norm_sq(roots)

    best_val = 1e10
    best_roots = None

    # Run Differential Evolution
    # Use a larger population and more iterations for robustness
    try:
        res = differential_evolution(
            de_cost,
            bounds,
            maxiter=1000,
            popsize=20,
            tol=1e-12,
            seed=42,
            polish=True  # This uses L-BFGS-B on the best result
        )
        
        if res.success or res.fun < best_val:
            best_roots = res.x
            # Calculate the actual score for the returned roots
            best_val = cost_func(best_roots)
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE failed or found a poor result, try a simple grid/random search fallback
    # to ensure we have a valid output
    if best_roots is None or best_val > 10.0:
        # Simple random search
        np.random.seed(42)
        for _ in range(1000):
            r = np.random.uniform(0.1, 10.0, 12)
            v = cost_func(r)
            if v < best_val:
                best_val = v
                best_roots = r

    # Final local polish on the best root set found
    if best_roots is not None and time.time() < deadline - 10:
        # Use L-BFGS-B directly for high precision
        def cost_log_roots(lr):
            r = np.exp(lr)
            return cost_func(r)
            
        log_roots = np.log(best_roots)
        try:
            res = minimize(
                cost_log_roots,
                log_roots,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
        except:
            pass

    # Fallback if all else fails
    if best_roots is None:
        best_roots = np.array([4.0 * (2.0 ** i) for i in range(12)])
        best_val = cost_func(best_roots)

    return list(best_roots)
