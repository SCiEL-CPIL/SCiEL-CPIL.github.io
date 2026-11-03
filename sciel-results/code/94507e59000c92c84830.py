"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
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
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation from monomial basis to Laguerre basis L_k^(1)
    # L_k^(1)(x) = sum_{j=0}^k a_{k,j} x^j
    # We need c such that sum c_k L_k^(1)(x) = sum p_j x^j
    # This means p = A^T c, where A[k,j] is coeff of x^j in L_k^(1)
    # So c = (A^T)^-1 p
    
    A = np.zeros((m+1, m+1))
    for k in range(m+1):
        for j in range(k+1):
            # L_k^(alpha)(x) = sum_{j=0}^k (-1)^j binom(k+alpha, k-j) x^j / j!
            # alpha=1
            # coeff of x^j is (-1)^j * binom(k+1, k-j) / j!
            # binom(k+1, k-j) = binom(k+1, j+1)
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # We want to solve A^T c = p for c
    # Let M = A^T. Then c = M^-1 p.
    # Precompute M^-1 or use solve.
    # Since p depends on roots, we compute p first, then solve.
    # To make it faster, we can precompute the inverse of A^T.
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial P(x) = prod(x-r_i) in the Laguerre basis.
        roots: list of 12 positive floats.
        """
        # Build polynomial coefficients p in monomial basis
        # p[0] is constant term, p[m] is coeff of x^m
        p = np.array([1.0])
        for r in roots:
            p = np.convolve(p, np.array([1.0, -r]))
        
        # Ensure p has length m+1
        if len(p) < m+1:
            p = np.pad(p, (0, m+1-len(p)))
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared = sum c_k^2 * (k+1) for n=1
        weights = np.arange(1, m+2) # k+1 for k=0..12
        norm_sq = np.dot(c**2, weights)
        return norm_sq

    def objective(y):
        """
        Objective function to minimize.
        We want to MAXIMIZE norm_sq, so we MINIMIZE -norm_sq.
        y: log of roots.
        """
        roots = np.exp(y)
        # Check for overflow/underflow or invalid roots
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e100
        try:
            nsq = compute_norm_sq(roots)
            if nsq <= 0:
                return 1e100
            return -nsq
        except Exception:
            return 1e100

    # Initial guesses
    # 1. Geometric progression (current best strategy)
    # 2. Roots of Laguerre polynomial
    # 3. Random positive numbers
    
    best_val = -1e100
    best_y = None

    # Start 1: Geometric progression optimized by seed
    # Seed was r0=4, q=2. Roots: 4, 8, 16...
    y_seed = np.log([4.0 * (2.0 ** i) for i in range(m)])
    
    # Start 2: Roots of L_12^(1)(x)
    # We can find roots of L_12^(1) using scipy or numpy
    # numpy doesn't have direct roots_laguerre for generalized, but we can use scipy.special.roots_genlaguerre
    from scipy.special import roots_genlaguerre
    roots_lag = roots_genlaguerre(m, n)
    y_lag = np.log(roots_lag.real)

    # Start 3: Uniformly spaced
    y_unif = np.log(np.linspace(0.1, 10.0, m))

    starts = [y_seed, y_lag, y_unif]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(10):
        y_rand = np.random.uniform(-2.0, 3.0, m) # log(0.1) to log(20)
        starts.append(y_rand)

    for y0 in starts:
        if not time_ok():
            break
        try:
            res = minimize(objective, y0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_y = res.x
        except Exception:
            continue

    # Polish with higher precision if time allows
    if best_y is not None and time_ok():
        try:
            res2 = minimize(objective, best_y, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                best_y = res2.x
        except Exception:
            pass

    if best_y is None:
        # Fallback
        return [4.0 * (2.0 ** i) for i in range(m)]

    best_roots = np.exp(best_y)
    # Sort roots? The order shouldn't matter for the polynomial, but let's keep them sorted for consistency
    best_roots = sorted(best_roots)
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
