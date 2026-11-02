# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_hermite
from scipy.optimize import minimize, differential_evolution
from itertools import product

def _get_hermite_values(n, x):
    """
    Compute the nth physicists' Hermite polynomial H_n(x) and its derivative H_n'(x)
    for an array x.
    H_n'(x) = 2n H_{n-1}(x)
    """
    if n == 0:
        Hn = np.ones_like(x)
        Hn_prime = np.zeros_like(x)
        return Hn, Hn_prime
    elif n == 1:
        Hn = 2 * x
        Hn_prime = 2 * np.ones_like(x)
        return Hn, Hn_prime
    
    H0 = np.ones_like(x)
    H1 = 2 * x
    H0_prime = np.zeros_like(x)
    H1_prime = 2 * np.ones_like(x)
    
    for k in range(2, n + 1):
        H2 = 2 * x * H1 - 2 * (k - 1) * H0
        H2_prime = 2 * k * H1
        
        H0, H1 = H1, H2
        H0_prime, H1_prime = H1_prime, H2_prime
        
    return H1, H1_prime

def _compute_bound(coeffs):
    """
    Compute the upper bound on C4 (uncertainty product) given coefficients [a, b, c].
    The H12 coefficient d is fixed by g(0) = 0.
    g(x) = (a H0 + b H4 + c H8 + d H12) * exp(-x^2/2)
    """
    a, b, c = coeffs
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0
    
    if not hasattr(_compute_bound, 'quad_data'):
        n_quad = 80
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0, dH0 = _get_hermite_values(0, x_quad)
        H4, dH4 = _get_hermite_values(4, x_quad)
        H8, dH8 = _get_hermite_values(8, x_quad)
        H12, dH12 = _get_hermite_values(12, x_quad)
        
        _compute_bound.quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12)
        
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12 = _compute_bound.quad_data
    
    P = a * H0 + b * H4 + c * H8 + d * H12
    dP = a * dH0 + b * dH4 + c * dH8 + d * dH12
    
    P2 = P * P
    norm2 = np.sum(w_quad * P2)
    
    if norm2 <= 1e-15:
        return 1e10
        
    var_x = np.sum(w_quad * x_quad**2 * P2) / norm2
    
    deriv_term = dP - x_quad * P
    deriv_term_sq = deriv_term * deriv_term
    norm_deriv2 = np.sum(w_quad * deriv_term_sq)
    
    var_k = norm_deriv2 / norm2
    
    bound = var_x * var_k
    
    return bound

def construct():
    """Find optimal coefficients [a, b, c] minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # The problem is homogeneous in scale. We can fix one variable or search in ratios.
    # Let's search in a bounded box. Since scale doesn't matter, we can restrict the norm.
    # However, differential_evolution works best with simple bounds.
    # Let's try a few different strategies.
    
    best_coeffs = np.array([1.0, 0.0, 0.0])
    best_score = _compute_bound(best_coeffs)
    
    # Strategy 1: Differential Evolution with a reasonable population
    # We use a time-check to stop if we exceed the deadline.
    
    try:
        # Use differential_evolution for global optimization
        # We'll run it with a callback to check time
        def callback(xk, convergence):
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
        
        # Bounds: The coefficients can be scaled arbitrarily.
        # We can constrain a=1 and search for b, c. But DE needs a box.
        # Let's search a, b, c in [-5, 5] and rely on the homogeneity.
        # Actually, it's better to fix one to avoid the flat direction.
        # Let's fix a=1 and search b, c.
        # But the interface requires returning 3 coefficients.
        # We can search [a, b, c] and then normalize.
        # Or search [b, c] with a=1.
        
        # Let's try searching [a, b, c] directly.
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        
        # Run DE
        result = differential_evolution(
            _compute_bound,
            bounds=bounds,
            maxiter=500,
            popsize=20,
            tol=1e-10,
            atol=1e-12,
            seed=42,
            polish=True,
            init='latinhypercube',
            callback=callback
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
            
    except Exception:
        pass
    
    # Strategy 2: Local refinement from multiple starting points
    # We can use the result from DE and also try some heuristic starting points.
    
    if time.time() < deadline - 10:
        # Generate some random starting points near the best found so far
        # and also some structured ones.
        
        starts = [best_coeffs.copy()]
        
        # Add some random perturbations
        for _ in range(10):
            pert = best_coeffs + np.random.randn(3) * 0.1
            starts.append(pert)
            
        # Add some structured starts
        starts.append(np.array([1.0, 0.1, 0.0]))
        starts.append(np.array([1.0, -0.1, 0.0]))
        starts.append(np.array([1.0, 0.0, 0.1]))
        starts.append(np.array([1.0, 0.0, -0.1]))
        
        for start in starts:
            if time.time() > deadline - 5:
                break
            
            try:
                res = minimize(
                    _compute_bound,
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x
            except Exception:
                pass
    
    # Ensure the returned coefficients are reasonable
    # The scale doesn't matter for the bound, but the evaluator might check validity.
    # The task says "construct() -> three coefficients".
    # We return the best found.
    
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
