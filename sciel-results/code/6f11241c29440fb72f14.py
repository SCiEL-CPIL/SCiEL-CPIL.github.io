"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_laguerre, factorial
from scipy.optimize import minimize, differential_evolution

# Constants for the problem
M = 12
N = 1

def _compute_score(roots):
    """
    Compute the upper bound C4 for a given set of 12 positive double-root locations.
    roots: array of 12 positive numbers (the double roots)
    Returns: the upper bound value (lower is better)
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0):
        return np.inf
    # Sort roots for consistency
    roots = np.sort(roots)
    
    # The Laguerre polynomial combination with double roots at each r_i
    # For m=12, n=1, we construct a polynomial P(x) = prod_{i=1}^{12} (x - r_i)^2
    # This is a polynomial of degree 24.
    # The Cohn-Goncalves construction relates to Fourier sign uncertainty.
    # The upper bound C4 is related to the maximum of |P(x)| * e^{-x/2} or similar.
    
    # Actually, for the refined uncertainty inequality with Laguerre polynomials,
    # the bound involves the Laguerre polynomial L_m^{(n)}(x) evaluated at the roots.
    # Let's think about this more carefully.
    
    # The Cohn-Goncalves bound: For a function with m sign changes, the uncertainty
    # product is bounded. The Laguerre polynomial construction gives an upper bound.
    
    # For m=12 double roots and n=1, the relevant quantity is likely:
    # C4 = max_x |L_12^{(1)}(x)| * e^{-x/2} * x^{1/2} or similar normalization
    
    # Actually, let me reconsider. The task says "12 double roots" and "dimension n=1".
    # The Laguerre polynomial L_m^{(alpha)}(x) has m distinct positive roots.
    # If we have "double roots", this suggests we're constructing a polynomial 
    # that has each of the 12 roots with multiplicity 2.
    
    # The upper bound on C4 for the Fourier sign uncertainty is likely:
    # C4 = (1/sqrt(pi)) * max_x |P(x)| * e^{-x/2} * x^{n/2}
    # where P(x) = prod_{i=1}^{12} (x - r_i)^2 is normalized somehow.
    
    # Let's use a standard formulation. The bound is:
    # C4 = max_x |prod_{i=1}^{12} (x - r_i)|^2 * e^{-x} * x^1 / (normalization)
    
    # A common formulation: C4 = max_x [prod_{i=1}^{12} (x - r_i)^2] * e^{-x} * x
    # But this needs proper normalization.
    
    # Let me try: the bound is the supremum of |f(x)|^2 e^{-x} x where
    # f is a polynomial with the given double roots, normalized so that
    # ∫ |f(x)|^2 e^{-x} x dx = 1.
    
    # For f(x) = prod_{i=1}^{12} (x - r_i), the normalization is:
    # ||f||^2 = ∫_0^∞ |f(x)|^2 e^{-x} x dx
    
    # But computing this integral exactly for degree-24 polynomials is complex.
    
    # Alternative: use the fact that for Laguerre polynomials, there are known
    # orthogonality relations. Let me try a direct numerical approach.
    
    # Let's compute the bound as:
    # C4 = max_x [ (prod_{i=1}^{12} (x - r_i)^2) * e^{-x} * x ]
    # This is a simple upper bound that doesn't require normalization.
    
    # Actually, for the Cohn-Goncalves bound, the relevant quantity is:
    # C4 = sup_x |P(x)|^2 e^{-x} x / ||P||_L2^2
    # where P(x) = prod (x - r_i) (not squared, since the square comes from |P|^2)
    
    # Let's compute this numerically.
    
    # First, compute the polynomial coefficients
    # P(x) = prod_{i=1}^{12} (x - r_i)
    # This is degree 12.
    
    # Compute coefficients using np.poly
    coeffs = np.poly(roots)  # This gives P(x) = prod (x - r_i), degree 12
    
    # Now we need: C4 = max_x |P(x)|^2 e^{-x} x / ||P||^2
    # where ||P||^2 = ∫_0^∞ |P(x)|^2 e^{-x} x dx
    
    # Let's compute the norm first using Gauss-Laguerre quadrature
    # Gauss-Laguerre with weight e^{-x}: ∫_0^∞ f(x) e^{-x} dx ≈ sum w_i f(x_i)
    # But we have weight e^{-x} * x, so we need ∫ P(x)^2 * x * e^{-x} dx
    # This is ∫ [P(x)^2 * x] e^{-x} dx, which we can do with standard GL quadrature.
    
    # Use high-order Gauss-Laguerre quadrature
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Evaluate P at quadrature points
    P_quad = np.polyval(coeffs, x_quad)
    
    # Compute norm squared: ∫ P(x)^2 * x * e^{-x} dx
    norm_sq = np.sum(w_quad * P_quad**2 * x_quad)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    # Now compute the maximum of |P(x)|^2 * x * e^{-x} / norm_sq
    # We need to find the maximum over x > 0.
    
    # Use a grid search + local optimization
    # The function g(x) = |P(x)|^2 * x * e^{-x} / norm_sq
    # is positive and goes to 0 as x→0 and x→∞, so it has a maximum.
    
    # Let's sample on a grid and then refine
    x_grid = np.linspace(0.01, 100, 5000)
    P_grid = np.polyval(coeffs, x_grid)
    g_grid = (P_grid**2 * x_grid * np.exp(-x_grid)) / norm_sq
    
    if not np.isfinite(np.max(g_grid)):
        return np.inf
    
    # Find the approximate location of the maximum
    idx_max = np.argmax(g_grid)
    x0 = x_grid[idx_max]
    
    # Refine with local optimization
    def neg_g(x):
        if x <= 0:
            return np.inf
        px = np.polyval(coeffs, x)
        return -(px**2 * x * np.exp(-x)) / norm_sq
    
    result = minimize(neg_g, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-12})
    
    if result.success and np.isfinite(result.fun):
        max_val = -result.fun
    else:
        max_val = np.max(g_grid)
    
    if not np.isfinite(max_val) or max_val < 0:
        return np.inf
    
    return max_val

def _objective(roots):
    """Objective function for optimization: the upper bound C4."""
    try:
        score = _compute_score(roots)
        if not np.isfinite(score):
            return 1e10
        return score
    except:
        return 1e10

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns: list of 12 positive numbers.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    # Initial guess: use the roots of the Laguerre polynomial L_12^{(1)}(x)
    # These are known to be positive and well-distributed.
    # The roots of L_m^{(alpha)}(x) can be found numerically.
    
    # Get the roots of the Laguerre polynomial L_12^{(1)}(x)
    # Using numpy's polynomial roots
    # L_12^{(1)}(x) coefficients
    from scipy.special import roots_laguerre
    x_lag, w_lag = roots_laguerre(12)
    # These are the roots of L_12^{(0)}(x). We need L_12^{(1)}.
    # For alpha=1, we can use:
    # Actually, roots_laguerre gives roots of L_n^{(0)}. 
    # For L_n^{(alpha)}, we need a different approach.
    
    # Let's just use the standard Laguerre roots as a starting point
    initial_roots = np.sort(x_lag)
    
    # Initial score
    best_roots = initial_roots.copy()
    best_score = _objective(best_roots)
    
    # Try differential evolution for global optimization
    # Bounds: each root should be positive. Let's use [0.01, 100]
    bounds = [(0.01, 100.0)] * 12
    
    def elapsed_ok():
        return (time.time() - start_time) < (deadline - 30)  # 30s safety margin
    
    # Run differential evolution
    try:
        if elapsed_ok():
            de_result = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='sobol',
                callback=lambda x, convergence: (time.time() - start_time < (deadline - 60))
            )
            if de_result.success and de_result.fun < best_score:
                best_roots = de_result.x
                best_score = de_result.fun
    except:
        pass
    
    # Local refinement with L-BFGS-B
    try:
        if elapsed_ok():
            local_result = minimize(
                _objective, best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if local_result.fun < best_score:
                best_roots = local_result.x
                best_score = local_result.fun
    except:
        pass
    
    # Additional local refinement with Nelder-Mead
    try:
        if elapsed_ok():
            nm_result = minimize(
                _objective, best_roots,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if nm_result.fun < best_score:
                best_roots = nm_result.x
                best_score = nm_result.fun
    except:
        pass
    
    # Ensure all roots are positive
    best_roots = np.maximum(best_roots, 0.01)
    
    # Sort the roots
    best_roots = np.sort(best_roots)
    
    return list(best_roots)

# EVOLVE-BLOCK-END
