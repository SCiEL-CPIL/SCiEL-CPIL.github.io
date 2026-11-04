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
    DEADLINE = 600.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the Laguerre basis transformation matrix
    # A[k, j] = coefficient of x^j in L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+1, k-j) * x^j / j!
    # = sum_{j=0}^{k} (-1)^j * binom(k+1, j+1) * x^j / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # We need c such that A^T @ c = p, where p are the standard basis coefficients
    # So c = (A^T)^{-1} @ p = solve(A.T, p)
    # Precompute the inverse of A.T for speed
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute the norm weights: w_k = (k+1) for n=1
    # ||P||^2 = sum_{k=0}^{12} c_k^2 * (k+1)
    w = np.array([k+1 for k in range(13)], dtype=float)

    def compute_coefficients_from_roots(roots):
        """Compute Laguerre coefficients from roots using Newton's identities or direct polynomial multiplication."""
        # Build polynomial P(x) = prod_{i=0}^{11} (x - r_i)
        # Using numpy polynomial construction
        p = np.array([1.0])
        for r in roots:
            p = np.convolve(p, np.array([1.0, -r]))
        # p has 13 coefficients: p[0] is constant term, p[12] is x^12 coefficient
        # Express in Laguerre basis: c = A_T_inv @ p
        c = A_T_inv @ p
        return c

    def cost_from_roots(roots):
        """Compute the upper bound on C4 from the 12 roots."""
        roots = np.array(roots)
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return 1e10
        c = compute_coefficients_from_roots(roots)
        norm_sq = np.dot(c**2, w)
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            return 1e10
        # The score is the upper bound on C4.
        # From the task context, the upper bound involves 1/||psi||^2 or similar.
        # Let's try: cost = 1/norm_sq (minimize this)
        # Or cost = norm_sq (minimize this)
        # The seed program used 1/norm_sq. Let's stick with that for consistency.
        return 1.0 / norm_sq

    # Strategy: Use the Dyson gas / mean-field equilibrium condition
    # For a linear confining potential V(x) = x, the equilibrium condition for
    # 12 particles with repulsion 2/(x_i - x_j) is:
    # sum_{j != i} 2/(x_i - x_j) = 1  (for each i)
    # This gives a system of 12 equations in 12 unknowns.
    # We can solve this system to get candidate roots, then verify and polish.

    def dyson_equilibrium_residuals(log_roots):
        """Residuals of the Dyson gas equilibrium condition in log-space."""
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return np.ones(12) * 1e10
        n_roots = 12
        residuals = np.zeros(n_roots)
        for i in range(n_roots):
            s = 0.0
            for j in range(n_roots):
                if i != j:
                    s += 2.0 / (roots[i] - roots[j])
            # Equilibrium: s = 1 (linear confining potential V'(x) = 1)
            residuals[i] = s - 1.0
        return residuals

    def dyson_cost(log_roots):
        """Cost function for Dyson gas optimization: minimize sum of squared residuals."""
        if not time_ok():
            return 1e10
        res = dyson_equilibrium_residuals(log_roots)
        if not np.all(np.isfinite(res)):
            return 1e10
        return np.sum(res**2)

    # Try to solve the Dyson gas equilibrium system
    best_value = 1e10
    best_roots = None

    # Initial guesses for log_roots
    # The roots should be positive and somewhat spread out.
    # For a linear potential, roots tend to be in [0, ~m] or similar.
    # Let's try several initial guesses.
    
    np.random.seed(42)
    
    # Guess 1: Logarithmically spaced in [0.1, 100]
    log_roots_init1 = np.log(np.linspace(0.1, 100, 12))
    
    # Guess 2: Linearly spaced in [0.1, 50]
    log_roots_init2 = np.log(np.linspace(0.1, 50, 12))
    
    # Guess 3: Around the expected scale
    log_roots_init3 = np.log(np.linspace(1.0, 30, 12))
    
    # Guess 4: Perturbed seed (r0=4, q=2)
    seed_roots = [4.0 * (2.0 ** i) for i in range(12)]
    log_roots_init4 = np.log(seed_roots)
    
    initial_guesses = [log_roots_init1, log_roots_init2, log_roots_init3, log_roots_init4]
    
    for guess in initial_guesses:
        if not time_ok():
            break
        try:
            result = minimize(dyson_cost, guess, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if np.isfinite(result.fun) and result.fun < 1e-6:
                log_roots_opt = result.x
                roots_opt = np.exp(log_roots_opt)
                if np.all(roots_opt > 0) and np.all(np.isfinite(roots_opt)):
                    val = cost_from_roots(roots_opt)
                    if val < best_value:
                        best_value = val
                        best_roots = roots_opt.copy()
        except Exception:
            continue
    
    # If Dyson gas solution didn't work well, fall back to direct optimization
    if best_roots is None or best_value > 1e5:
        # Direct optimization in log-space with 12 DOF
        def direct_cost(log_roots):
            if not time_ok():
                return 1e10
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
                return 1e10
            return cost_from_roots(roots)
        
        # Multiple starts
        for _ in range(20):
            if not time_ok():
                break
            # Random initial guess in log-space
            log_roots_init = np.random.uniform(-2.0, 4.0, 12)
            try:
                result = minimize(direct_cost, log_roots_init, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                if np.isfinite(result.fun) and result.fun < best_value:
                    log_roots_opt = result.x
                    roots_opt = np.exp(log_roots_opt)
                    if np.all(roots_opt > 0) and np.all(np.isfinite(roots_opt)):
                        val = cost_from_roots(roots_opt)
                        if val < best_value:
                            best_value = val
                            best_roots = roots_opt.copy()
            except Exception:
                continue
    
    # Final polish: if we have a good candidate, do a high-precision local optimization
    if best_roots is not None and time_ok():
        def final_cost(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
                return 1e10
            return cost_from_roots(roots)
        
        try:
            result = minimize(final_cost, np.log(best_roots), method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if np.isfinite(result.fun) and result.fun < best_value:
                log_roots_final = result.x
                roots_final = np.exp(log_roots_final)
                if np.all(roots_final > 0) and np.all(np.isfinite(roots_final)):
                    val = cost_from_roots(roots_final)
                    if val < best_value:
                        best_value = val
                        best_roots = roots_final.copy()
        except Exception:
            pass
    
    # If all else fails, use the seed
    if best_roots is None:
        best_roots = np.array([4.0 * (2.0 ** i) for i in range(12)])
    
    # Ensure roots are positive and finite
    best_roots = np.maximum(best_roots, 1e-10)
    best_roots = best_roots[np.isfinite(best_roots)]
    
    # If we lost some roots, pad with reasonable values
    while len(best_roots) < 12:
        best_roots = np.append(best_roots, best_roots[-1] * 1.1)
    
    return list(best_roots[:12])
# EVOLVE-BLOCK-END
