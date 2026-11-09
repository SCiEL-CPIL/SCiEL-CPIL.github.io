"""
Strategy: Log-Spaced Root Optimization for Cohn-Goncalves Construction
Instead of the expensive and unstable analytical Fourier transform integration 
used in the previous strategy, we hypothesize that the optimal root distribution 
for minimizing the uncertainty constant (or maximizing the sign change location 
relative to the decay) is well-approximated by a geometric progression.
This reduces the 12-dimensional optimization problem to a 2-parameter search 
(start root r1 and ratio rho), which can be solved extremely fast and robustly.
We optimize a surrogate objective that penalizes small first sign changes 
and rewards rapid decay (large roots), balancing the two competing requirements 
of the uncertainty bound.
"""
import numpy as np
import time
from scipy.optimize import minimize
import sys

# EVOLVE-BLOCK-START

def _surrogate_objective(params):
    """
    Surrogate objective for log-spaced roots.
    params: [r1, log_rho]
    We use log_rho to ensure positivity and smooth optimization.
    
    The C4 bound is roughly proportional to 1 / (decay_rate * sign_change_location).
    The decay rate is determined by the largest root (or the envelope).
    The sign change location is determined by the smallest root (or the first zero).
    
    For a Laguerre-like polynomial with double roots r_i, the first sign change 
    of the Fourier transform is roughly related to the smallest root.
    The decay in the time domain (or frequency, depending on convention) is 
    related to the spread of roots.
    
    A simple heuristic: 
    Objective = - (log(first_sign_change) + log(decay_factor))
    We want to maximize this.
    
    Let's approximate:
    r_i = r1 * exp((i-1)*log_rho)
    
    The "effective" first sign change X_c is approximated by the smallest root r_min = r1.
    The "effective" decay rate is approximated by the largest root r_max = r1 * rho^11.
    To keep the function well-behaved (Laguerre weight e^{-x/2}), roots shouldn't be too large.
    However, larger roots generally push the sign change out and increase decay.
    
    Let's define the score as:
    Score = X_sign * Decay
    where X_sign ~ r1 (first root)
    Decay ~ r_max (largest root)
    But we must ensure the roots are distinct and positive.
    
    Actually, the Cohn-Goncalves bound C4 is often bounded by a function of the 
    ratio of the integral of the function to the integral of its derivative or similar.
    For the specific "sign uncertainty" task, the critical quantity is the 
    location of the first zero of the Fourier transform.
    
    Let's use a more robust surrogate:
    We want to maximize the "spread" of the roots to increase the decay, 
    while keeping the first root large enough to delay the first sign change.
    However, if roots are too spread out, the polynomial oscillates wildly.
    
    Let's stick to a simple proxy:
    Maximize: min(root) * max(root)
    Subject to: roots > 0, and perhaps a penalty if the ratio is too extreme.
    
    Let's try:
    f(r1, log_rho) = r1 * (r1 * exp(11 * log_rho)) = r1^2 * exp(11 * log_rho)
    This suggests making r1 and rho as large as possible. 
    But there is a constraint: the function must be "Laguerre-like".
    Usually, the optimal roots for minimizing the L1 norm of the Fourier 
    transform of a polynomial with fixed degree are **equally spaced** in 
    a specific metric.
    
    Let's try a different surrogate:
    The upper bound on C4 is often related to the inverse of the "width" 
    of the main lobe of the Fourier transform.
    The width is inversely proportional to the range of the roots.
    So, maximize (r_max - r_min).
    
    Let's optimize:
    Objective = - (r_max - r_min)
    This is too simple and will push roots to infinity.
    
    We need a balance. The Laguerre weight e^{-x/2} penalizes large x.
    The effective contribution of a root at x is weighted by e^{-x/2}.
    So, we want to maximize the weighted spread.
    
    Let's define the objective as minimizing:
    - sum_{i} r_i * exp(-r_i/2)
    This encourages roots to be in the "sweet spot" where the weight is significant 
    but the roots are large enough to provide decay.
    """
    r1 = params[0]
    log_rho = params[1]
    
    if r1 <= 0:
        return 1e9
        
    rho = np.exp(log_rho)
    
    # Generate roots
    # r_i = r1 * rho^(i-1) for i=1..12
    exponents = np.arange(12)
    roots = r1 * (rho ** exponents)
    
    # Check validity
    if np.any(roots <= 0) or np.any(np.isinf(roots)) or np.any(np.isnan(roots)):
        return 1e9
        
    # Surrogate: Minimize negative weighted sum of roots
    # The weight e^{-x/2} reflects the Laguerre decay.
    weights = np.exp(-roots / 2.0)
    weighted_sum = np.sum(roots * weights)
    
    # We want to MAXIMIZE weighted_sum to push roots into the high-impact region.
    # So we minimize -weighted_sum.
    return -weighted_sum

def construct():
    deadline = time.time() + 600.0  # 600s limit, keep margin
    
    # Initial guess for r1 and log_rho
    # Typical roots for m=12 might be around 1 to 100.
    # Let's start with r1 = 2.0, rho = 1.2 (roots: 2, 2.4, ..., ~13)
    # Or r1 = 1.0, rho = 1.5 (roots: 1, 1.5, ..., ~143)
    
    # Let's try a few initial guesses
    initial_guesses = [
        [2.0, np.log(1.2)],
        [1.0, np.log(1.5)],
        [5.0, np.log(1.1)],
        [10.0, np.log(1.05)]
    ]
    
    best_params = initial_guesses[0]
    best_score = _surrogate_objective(best_params)
    
    bounds = [(0.1, 100.0), (-2.0, 2.0)]  # r1 > 0, log_rho in reasonable range
    
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                _surrogate_objective,
                guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract roots
    r1, log_rho = best_params
    rho = np.exp(log_rho)
    exponents = np.arange(12)
    roots = r1 * (rho ** exponents)
    
    # Ensure strict positivity and ordering
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-6)
    for i in range(1, 12):
        if roots[i] <= roots[i-1]:
            roots[i] = roots[i-1] + 1e-6
            
    return [float(r) for r in roots]

# EVOLVE-BLOCK-END
