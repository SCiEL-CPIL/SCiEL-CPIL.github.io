"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import eval_hermite
import math

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0 # 180s limit + safety margin

def _hermite_polynomial(n, x):
    """Evaluate Hermite polynomial H_n(x) using scipy."""
    return eval_hermite(n, x)

def _compute_coefficients(a, b, c):
    """
    Compute the four coefficients [a, b, c, d] for H0, H4, H8, H12.
    Constraint: g(0) = 0 => P(0) = 0.
    H_n(0) for even n:
    H_0(0) = 1
    H_4(0) = -3
    H_8(0) = 105
    H_12(0) = -10395
    Equation: a*1 + b*(-3) + c*105 + d*(-10395) = 0
    d = (a - 3b + 105c) / 10395
    """
    d = (a - 3.0 * b + 105.0 * c) / 10395.0
    return np.array([a, b, c, d])

def _evaluate_g_and_fg(coeffs, x):
    """
    Evaluate g(x) and its Fourier transform coefficients at x.
    g(x) = (a H0 + b H4 + c H8 + d H12) * exp(-x^2/2)
    F[g](xi) = (a (-i)^0 H0 + b (-i)^4 H4 + c (-i)^8 H8 + d (-i)^12 H12) * exp(-xi^2/2)
    Note: (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
    So the Fourier transform has the SAME polynomial structure, just evaluated at xi.
    Wait, check the transform convention.
    If F[f](k) = int f(x) exp(-i k x) dx, then F[H_n(x) exp(-x^2/2)] = (-i)^n H_n(k) exp(-k^2/2).
    For n=0,4,8,12, (-i)^n is always 1.
    So F[g](k) = (a H0(k) + b H4(k) + c H8(k) + d H12(k)) * exp(-k^2/2).
    Thus g(x) and F[g](x) have the exact same functional form!
    This simplifies things greatly. ||g||_1 == ||F[g]||_1 and ||g||_2 == ||F[g]||_2.
    Therefore, the objective function simplifies to:
    Score = (||g||_1 * ||F[g]||_1) / (||g||_2 * ||F[g]||_2) = (||g||_1^2) / (||g||_2^2)
    """
    a, b, c, d = coeffs
    # Calculate polynomial part P(x)
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 42x^10 + 1260x^8 - 16800x^6 + 113400x^4 - 226800x^2 + 10395
    
    # Using eval_hermite is safer for accuracy
    P = a * _hermite_polynomial(0, x) + \
        b * _hermite_polynomial(4, x) + \
        c * _hermite_polynomial(8, x) + \
        d * _hermite_polynomial(12, x)
        
    g = P * np.exp(-x**2 / 2.0)
    return g

def _objective(params):
    """
    Objective function to minimize.
    params: [a, b, c]
    Returns: (||g||_1^2) / (||g||_2^2)
    """
    a, b, c = params
    coeffs = _compute_coefficients(a, b, c)
    
    # Grid for numerical integration
    # The function decays as exp(-x^2/2). A range of [-10, 10] is sufficient.
    # We need high resolution for L1 norm due to potential oscillations/sign changes.
    x = np.linspace(-10, 10, 10000)
    g = _evaluate_g_and_fg(coeffs, x)
    
    # Calculate norms
    # L2 norm squared is integral of g^2
    l2_sq = np.trapz(g**2, x)
    if l2_sq <= 1e-12:
        return 1e12 # Avoid division by zero, return large penalty
    
    # L1 norm is integral of |g|
    l1 = np.trapz(np.abs(g), x)
    
    # Objective: L1^2 / L2^2
    # Note: Since F[g] has same norms, the ratio is (L1_g * L1_Fg) / (L2_g * L2_Fg) = L1^2 / L2^2
    return (l1 * l1) / l2_sq

def construct():
    """
    Returns the three coefficients [a, b, c] that minimize the uncertainty bound.
    """
    # Seed values from the current program
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_coeffs.copy()
    best_score = _objective(best_params)
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT * 0.5:
        return best_params

    # Define bounds for a, b, c
    # The seed values are roughly -1, 0.04, 0.0003.
    # We can expand the search space significantly to find global minimum.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use Differential Evolution
    # Tolerances and iterations tuned for speed and accuracy
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-8,
            polish=True
        )
        
        if result.success or result.fun < best_score:
            best_params = result.x
            best_score = result.fun
            
    except Exception as e:
        # If optimization fails, return seed
        pass
        
    # Final check
    if time.time() - START_TIME > TIME_LIMIT * 0.8:
        return best_params

    # Refine with Nelder-Mead if time permits
    try:
        from scipy.optimize import minimize
        res = minimize(_objective, best_params, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass

    return best_params
# EVOLVE-BLOCK-END
