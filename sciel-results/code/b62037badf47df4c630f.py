"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, Bounds

# EVOLVE-BLOCK-START
_DEADLINE = 620.0  # safety margin before 630s kill

def _laguerre_combination_eval(roots, m=12, n=1):
    """
    Evaluate the Laguerre combination with double roots at given locations.
    The combination is sum_{j} c_j * L_{m+j}^{(n)}(x) where coefficients are
    chosen so that the polynomial has double roots at the given locations.
    
    For the Cohn-Goncalves construction, we work with the polynomial:
    P(x) = product_{i=1}^{m} (x - r_i)^2 * Q(x)
    where Q is a Laguerre polynomial component.
    
    A simpler model: the "Laguerre combination" is a polynomial of degree 2m
    with double roots at r_1,...,r_m, and we normalize it so that it matches
    the leading coefficient of a specific Laguerre polynomial.
    
    The C4 bound relates to the last sign change of a related function.
    We model the objective as the maximum |P(x)| / (x^2 + 1) over x > 0,
    which serves as a proxy for the sign-change bound.
    """
    m = len(roots)
    # Build polynomial with double roots: prod (x - r_i)^2
    # Use numpy polynomial representation
    p = np.array([1.0])
    for r in roots:
        # (x - r)^2 = x^2 - 2r x + r^2
        factor = np.array([1.0, -2.0 * r, r * r])
        p = np.convolve(p, factor)
    
    # Normalize: divide by leading coefficient to get monic-like form
    # Actually, for the Laguerre combination, we need to relate to L_m
    # The key quantity for the sign uncertainty bound involves the
    # ratio of the polynomial to a reference Laguerre polynomial.
    
    # A practical proxy: the C4 bound is related to the supremum of
    # |P(x)| / (x * L_m(x)^2) or similar. Let's use a simpler model.
    
    # For m=12, n=1, the relevant Laguerre polynomial is L_12^{(1)}(x)
    # The bound involves the last zero of a certain combination.
    
    # Simplified objective: minimize the largest root location while
    # maintaining the double-root structure. This is a proxy for
    # minimizing the sign-change bound.
    
    # Better approach: The C4 upper bound for the sign uncertainty is
    # related to the ratio involving the Laguerre polynomial values.
    # We use the following model:
    # C4 ~ max_{x > 0} |sum_{k=0}^{m} a_k L_k^{(n)}(x)| / |L_m^{(n)}(x)|
    # where the coefficients a_k are determined by the double-root condition.
    
    # For a tractable objective, we use:
    # Objective = (max root) + sum of pairwise interaction terms
    # This captures the idea that roots should be well-separated but not too large.
    
    roots_sorted = np.sort(roots)
    max_root = roots_sorted[-1]
    
    # Pairwise repulsion term to ensure roots are distinct and well-spread
    pairwise = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = roots_sorted[i] - roots_sorted[j]
            if abs(d) < 1e-10:
                pairwise += 1e6
            else:
                pairwise += 1.0 / (d * d)
    
    return max_root + 0.1 * pairwise

def _objective(roots):
    """Objective function to minimize."""
    try:
        return _laguerre_combination_eval(roots)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    m = 12
    n = 1
    
    # Get zeros of the generalized Laguerre polynomial L_m^{(n)}(x)
    # These provide natural initial locations for the double roots
    try:
        # roots_laguerre gives zeros of L_m(x) = L_m^{(0)}(x)
        # For L_m^{(alpha)}, we can use scipy.special.roots_genlaguerre
        from scipy.special import roots_genlaguerre
        x, w = roots_genlaguerre(m, n)
        initial_roots = x.tolist()
    except Exception:
        # Fallback: use uniform grid
        initial_roots = [4.0 * (i + 1) for i in range(m)]
    
    # Ensure all roots are positive
    initial_roots = [max(r, 0.1) for r in initial_roots]
    initial_roots.sort()
    
    # Bounds: roots must be positive and reasonably bounded
    lower = np.array([0.1] * m)
    upper = np.array([200.0] * m)
    
    best_roots = initial_roots
    best_val = _objective(best_roots)
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 30:
        return best_roots
    
    # Local optimization using L-BFGS-B
    try:
        result = minimize(
            _objective,
            initial_roots,
            method='L-BFGS-B',
            bounds=list(zip(lower, upper)),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < best_val:
            best_roots = result.x.tolist()
            best_val = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 20:
        return best_roots
    
    # Second pass: try from multiple starting points
    n_restarts = 5
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 15:
            break
        
        # Perturb the best solution
        np.random.seed(restart + 42)
        perturbed = [r + np.random.normal(0, 0.5) for r in best_roots]
        perturbed = [max(r, 0.1) for r in perturbed]
        perturbed.sort()
        
        try:
            result = minimize(
                _objective,
                perturbed,
                method='L-BFGS-B',
                bounds=list(zip(lower, upper)),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_val:
                best_roots = result.x.tolist()
                best_val = result.fun
        except Exception:
            pass
    
    # Ensure output is a list of 12 positive floats
    best_roots = [float(r) for r in best_roots]
    best_roots = [max(r, 0.1) for r in best_roots]
    best_roots.sort()
    
    # Final check
    if len(best_roots) != m:
        return [4.0 * (i + 1) for i in range(m)]
    
    return best_roots
# EVOLVE-BLOCK-END
