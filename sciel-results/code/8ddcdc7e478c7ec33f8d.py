"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
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
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 30.0  # return 30s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

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
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(params):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        a, log_q = params
        
        # Ensure a is positive and q is positive
        # a > 0, q > 0
        # We optimize in log-space for q to ensure positivity and cover wide range.
        # a can be optimized directly or in log-space. Let's use log_a for symmetry.
        # params = [log_a, log_q]
        
        log_a, log_q = params
        a = math.exp(log_a)
        q = math.exp(log_q)
        
        # Generate roots: r_i = a * q^(i) for i=0..11
        # The seed was r0=4, q=2 -> roots 4, 8, ...
        # Let's generate 12 roots.
        i = np.arange(12)
        roots = a * (q ** i)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over (log_a, log_q) using Differential Evolution.
    # 2. Local refinement using L-BFGS-B.
    
    # Seed: a=4, q=2
    # log_a = log(4) ~ 1.386
    # log_q = log(2) ~ 0.693
    
    # Bounds:
    # a can vary. If a is very small, roots are small. If a is very large, roots are large.
    # q can vary. If q is close to 1, roots are close together. If q is large, they spread.
    # Let's set wide bounds in log-space.
    # log_a in [-5, 5] -> a in [0.006, 148]
    # log_q in [-2, 3] -> q in [0.13, 20]
    
    bounds = [(-5.0, 5.0), (-2.0, 3.0)]
    
    best_roots = None
    best_val = 1e10
    
    # Run Differential Evolution
    # popsize=20, maxiter=100 is usually fast for 2D.
    # We have 600s, so we can be generous.
    
    try:
        # Use seed for reproducibility if needed, but DE is stochastic.
        # Let's try a few restarts or just one good run.
        result = differential_evolution(
            cost_func, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=25, 
            tol=1e-10, 
            polish=True,  # L-BFGS-B polish
            init='sobol' # Better initial distribution
        )
        
        if result.success or result.fun < best_val:
            best_val = result.fun
            log_a, log_q = result.x
            a = math.exp(log_a)
            q = math.exp(log_q)
            best_roots = a * (q ** np.arange(12))
            
    except Exception as e:
        pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    # The seed from the prompt: r0=4, q=2
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func([math.log(4.0), math.log(2.0)])
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots

    # Additional Local Polish from multiple starting points to ensure we found the global min
    # in the 2D space.
    start_points = [
        [math.log(4.0), math.log(2.0)], # Seed
        [0.0, 0.0],                     # a=1, q=1 (degenerate)
        [1.0, 0.5],                     # a=e, q=sqrt(e)
        [-1.0, 1.0],                    # a=1/e, q=e
        [2.0, 0.0],                     # a=e^2, q=1
        [0.0, 1.0]                      # a=1, q=e
    ]
    
    for sp in start_points:
        if not time_ok():
            break
        try:
            res = minimize(cost_func, sp, method='L-BFGS-B', bounds=bounds)
            if res.fun < best_val:
                best_val = res.fun
                log_a, log_q = res.x
                a = math.exp(log_a)
                q = math.exp(log_q)
                best_roots = a * (q ** np.arange(12))
        except Exception:
            continue

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
