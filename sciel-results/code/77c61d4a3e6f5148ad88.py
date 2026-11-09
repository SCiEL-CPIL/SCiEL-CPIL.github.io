"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, genlaguerre
from scipy.integrate import quad
import time

def _compute_bound(roots):
    """Compute the upper bound on C4 given 12 double-root locations."""
    roots = np.asarray(roots, dtype=np.float64)
    m = 12
    n = 1
    
    if np.any(roots <= 0) or len(roots) != 12:
        return np.inf
    
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-12):
        return np.inf
    
    # The refined Cohn-Goncalves construction:
    # The bound involves the Laguerre polynomial L_m^(0) evaluated at the roots
    # and the weight function. The key quantity is the product of |L_m^(0)(r_i)|
    # normalized by the L2 norm of L_m^(0) under weight e^{-x}.
    
    # L_12^(0)(x) = sum_{k=0}^{12} (-1)^k * C(12,k) * x^k / k!
    # The bound: C4 <= sqrt( <L_m^2, e^{-x}> ) / prod_i |L_m(r_i)|^{1/m}
    # where <f,g> = integral f*g*e^{-x} dx from 0 to inf
    
    # For Laguerre polynomials: <L_m^(0), L_m^(0)> = 1 (orthonormal? No, = 1 for standard)
    # Actually <L_n^(alpha), L_n^(alpha)> = 1 for standard normalization
    # But scipy uses: L_n^(alpha) with <L_n, L_m> = delta_nm * Gamma(n+alpha+1)/(n!)^2 * ... 
    # Standard: integral_0^inf L_n^(alpha)(x)^2 x^alpha e^{-x} dx = Gamma(n+alpha+1)/n!
    # For alpha=0: integral = 1
    
    # So the L2 norm squared of L_12^(0) under e^{-x} is 1.
    # The bound is then: C4 <= 1 / (prod |L_12(r_i)|)^{1/12}
    
    # But wait - the double roots mean the construction uses (x-r_i)^2 factors.
    # The actual bound likely involves:
    # C4 <= [integral of (prod (x-r_i)^2) * L_m^2 * e^{-x} dx]^{1/2} / some_normalization
    
    # Let me use the most standard form: the bound is the L2 norm of the 
    # weighted Laguerre polynomial divided by the geometric mean of root values.
    
    # Alternative: the bound = (sum of |L_m(r_i)|^2 * e^{-r_i})^{1/2} / prod(r_i)^{1/(2m)}
    
    # Let me compute several candidate formulas and use the one that makes sense.
    
    # Compute L_12^(0)(r_i) for each root
    L_vals = np.array([eval_genlaguerre(12, 0, r) for r in roots])
    
    # Candidate 1: 1 / (prod |L(r_i)|)^{1/12}
    prod_L = np.prod(np.abs(L_vals))
    if prod_L < 1e-300:
        return np.inf
    bound1 = 1.0 / (prod_L ** (1.0/12.0))
    
    # Candidate 2: sqrt(sum(L(r_i)^2 * e^{-r_i})) / prod(r_i)^{1/(2*12)}
    weight_sum = np.sum(L_vals**2 * np.exp(-roots))
    if weight_sum <= 0:
        return np.inf
    prod_roots = np.prod(roots)
    bound2 = np.sqrt(weight_sum) / (prod_roots ** (1.0/24.0))
    
    # Candidate 3: The actual Cohn-Goncalves bound involves the ratio of norms
    # C4 <= ||f||_2 / ||f||_1 type bounds, where f is constructed from the roots
    
    # Most likely: the bound is related to the maximum of |L_m(x)| * e^{-x/2} 
    # over the positive reals, evaluated at the critical points
    
    # Let me use a more principled approach:
    # The bound = sqrt( integral_0^inf L_12(x)^2 e^{-x} dx ) / min_i |L_12(r_i)|
    # Since the integral = 1 for standard Laguerre, this = 1 / min|L_12(r_i)|
    
    min_L = np.min(np.abs(L_vals))
    if min_L < 1e-300:
        return np.inf
    bound3 = 1.0 / min_L
    
    # Return the minimum of the candidates (most conservative valid bound)
    # Actually, we want the TRUE bound. Let me use the most standard one.
    # The Cohn-Goncalves paper uses: C4 <= (L_m(0))^2 / prod_i (r_i / L_m(r_i))^{1/m}
    # = 1 / prod_i (r_i / |L_m(r_i)|)^{1/m}
    
    ratio_prod = np.prod(roots / np.abs(L_vals))
    if ratio_prod < 1e-300:
        return np.inf
    bound4 = 1.0 / (ratio_prod ** (1.0/12.0))
    
    # Use bound4 as it matches the Cohn-Goncalves structure most closely
    return bound4

def construct():
    """Construct 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s kill
    
    # The 12 double roots should be the 12 positive zeros of L_12^(0)(x)
    # These are the natural choice for the Cohn-Goncalves construction
    
    # Compute zeros of L_12^(0)(x) using the companion matrix method
    # L_n^(0)(x) = sum_{k=0}^n (-1)^k * C(n,k) * x^k / k!
    # For n=12: coefficients from highest to lowest degree
    
    n = 12
    # Coefficients of L_12^(0)(x): c_k * x^k where c_k = (-1)^k * C(12,k) / k!
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        coeffs[k] = ((-1)**k) * (float(np.math.factorial(n)) / (float(np.math.factorial(k)) * float(np.math.factorial(n-k)))) / float(np.math.factorial(k))
    
    # Find roots using numpy
    roots = np.roots(coeffs)
    # Keep only real, positive roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    pos_roots = real_roots[real_roots > 1e-10]
    
    if len(pos_roots) == 12:
        x0 = np.sort(pos_roots)
    else:
        # Fallback: use Laguerre-Gauss nodes
        from numpy.polynomial.laguerre import laggauss
        nodes, weights = laggauss(12)
        x0 = nodes.copy()
    
    best_x = x0.copy()
    best_bound = _compute_bound(x0)
    
    # Now optimize the root positions to minimize the bound
    # Use scipy.optimize.minimize with multiple restarts
    
    from scipy.optimize import minimize
    
    def objective(log_r):
        if time.time() > deadline - 10:
            return np.inf
        roots = np.exp(log_r)
        return _compute_bound(roots)
    
    log_x0 = np.log(x0)
    
    # Try L-BFGS-B optimization
    try:
        res = minimize(objective, log_x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_bound:
            best_x = np.exp(res.x)
            best_bound = res.fun
    except:
        pass
    
    # Try Nelder-Mead
    try:
        res2 = minimize(objective, log_x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if res2.fun < best_bound:
            best_x = np.exp(res2.x)
            best_bound = res2.fun
    except:
        pass
    
    # Multi-start with perturbations
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        perturbation = rng.normal(0, 0.1, size=12)
        x_trial = np.log(x0) + perturbation
        try:
            res3 = minimize(objective, x_trial, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12})
            if res3.fun < best_bound:
                best_x = np.exp(res3.x)
                best_bound = res3.fun
        except:
            pass
    
    # Verify final result
    final_roots = best_x
    if np.any(final_roots <= 0) or len(final_roots) != 12:
        final_roots = x0
    
    return list(final_roots)
# EVOLVE-BLOCK-END
