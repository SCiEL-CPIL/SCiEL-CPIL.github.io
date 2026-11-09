"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre, factorial
from scipy.optimize import minimize, differential_evolution

M = 12  # number of double roots
N = 1   # dimension parameter

def _laguerre_k(x, k, alpha):
    """Generalized Laguerre polynomial L_k^{(alpha)}(x)."""
    return eval_genlaguerre(k, alpha, x)

def _compute_bound(roots):
    """
    Compute the upper bound C4 for the given set of positive double roots.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != M:
        return np.inf
    if np.any(roots <= 0):
        return np.inf
    if not np.all(np.diff(np.sort(roots)) > 1e-15):
        return np.inf
    
    roots_sorted = np.sort(roots)
    alpha = 2 * N - 1  # alpha = 1
    
    # Compute the L2 norm of P(x) = prod (x - r_i)^2 with weight x^alpha * e^{-x}
    # Using the Laguerre expansion: P(x) = sum c_k L_k^{(alpha)}(x)
    # ||P||^2 = sum c_k^2 * ||L_k^{(alpha)}||^2 where ||L_k^{(alpha)}||^2 = Gamma(k+alpha+1)/k!
    
    # First, build the polynomial P(x) = prod (x - r_i)^2
    # Then express it in the Laguerre basis
    
    # Coefficients of P(x) in monomial basis
    # P(x) = prod_{i=1}^{m} (x - r_i)^2
    # Let's compute the coefficients
    
    # Start with [1] (the constant 1)
    coeffs = np.array([1.0])
    for r in roots_sorted:
        # Multiply by (x - r)^2 = x^2 - 2r*x + r^2
        new_coeffs = np.zeros(len(coeffs) + 2)
        new_coeffs[2:] += coeffs  # x^2 * coeffs
        new_coeffs[1:-1] += -2*r * coeffs  # -2r*x * coeffs
        new_coeffs[0:-2] += r**2 * coeffs  # r^2 * coeffs
        coeffs = new_coeffs
    
    # coeffs[0] is the constant term, coeffs[-1] is the leading coefficient
    # P(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[-1]*x^{2m}
    
    # Now express P(x) in Laguerre basis L_k^{(alpha)}(x)
    # L_k^{(alpha)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+alpha, k-j) * x^j / j!
    # Leading term of L_k^{(alpha)}(x) is (-1)^k * binom(k+alpha, k) * x^k / k!
    # = (-1)^k * (k+alpha)! / (alpha! * k!) * x^k / k!
    
    # To find c_k in P(x) = sum c_k L_k^{(alpha)}(x):
    # Use the fact that the leading coefficient of P(x) is coeffs[-1] = 1 (monic)
    # And the leading coefficient of L_k^{(alpha)}(x) is (-1)^k * Gamma(k+alpha+1) / (Gamma(alpha+1) * (k!)^2)
    
    # Actually, let's use a different approach: evaluate the norm directly
    # ||P||^2 = int_0^inf P(x)^2 * x^alpha * e^{-x} dx
    # This can be computed by expanding P(x)^2 in monomials and using:
    # int_0^inf x^n * e^{-x} dx = Gamma(n+1) = n!
    
    # P(x)^2 has degree 4m = 48
    # Compute coefficients of P(x)^2
    p2_coeffs = np.convolve(coeffs, coeffs)
    
    # ||P||^2 = sum_{n=0}^{4m} p2_coeffs[n] * Gamma(n+1)
    # But wait, P(x) = sum_{n=0}^{2m} coeffs[n] * x^n
    # P(x)^2 = sum_{n=0}^{4m} p2_coeffs[n] * x^n
    # int_0^inf x^n * x^alpha * e^{-x} dx = Gamma(n+alpha+1)
    
    norm_sq = 0.0
    for n in range(len(p2_coeffs)):
        norm_sq += p2_coeffs[n] * math.gamma(n + alpha + 1)
    
    # The bound C4 is related to 1/norm_sq or norm_sq itself
    # For the Cohn-Goncalves construction, the bound on the number of sign changes
    # involves the reciprocal of this norm or a related quantity.
    
    # Let's return the norm as the bound (to be minimized)
    return norm_sq

def construct():
    """Find optimal 12 positive double-root locations."""
    deadline = time.time() + 600
    
    # Strategy: Use differential evolution to minimize the bound
    # But first, let's try some analytical approaches
    
    # Approach 1: Zeros of Laguerre polynomial L_m^{(alpha)}(x)
    try:
        n_zeros = M
        alpha_val = 2 * N - 1  # = 1
        
        d = np.array([2*i + alpha_val - 1 for i in range(n_zeros)])
        e = np.array([math.sqrt((i+1)*(i+1+alpha_val)) for i in range(n_zeros-1)])
        
        J = np.zeros((n_zeros, n_zeros))
        for i in range(n_zeros):
            J[i, i] = d[i]
        for i in range(n_zeros-1):
            J[i, i+1] = e[i]
            J[i+1, i] = e[i]
        
        zeros = np.sort(np.linalg.eigvalsh(J))
        if np.all(zeros > 0) and len(zeros) == M:
            bound_laguerre = _compute_bound(zeros)
            seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
            bound_seed = _compute_bound(seed_roots)
            # If Laguerre zeros give a better bound, use them
            if bound_laguerre < bound_seed:
                return [float(z) for z in zeros]
    except Exception:
        pass
    
    # Approach 2: Try to optimize using differential evolution with a time limit
    try:
        def objective(x):
            if time.time() > deadline - 30:
                return np.inf
            if np.any(x <= 0):
                return np.inf
            if not np.all(np.diff(np.sort(x)) > 1e-10):
                return np.inf
            return _compute_bound(x)
        
        # Bounds for each root
        bounds = [(0.1, 100.0)] * M
        
        # Use differential evolution with limited iterations
        result = differential_evolution(
            objective, bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True
        )
        
        if result.success and result.fun < np.inf:
            best_roots = np.sort(result.x)
            if np.all(best_roots > 0) and np.all(np.diff(best_roots) > 1e-10):
                return [float(r) for r in best_roots]
    except Exception:
        pass
    
    # Fallback: seed values
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
