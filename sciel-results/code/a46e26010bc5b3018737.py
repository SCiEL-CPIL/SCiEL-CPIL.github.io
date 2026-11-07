"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import roots_hermite

# Pre-compute Hermite polynomial coefficients for H0, H4, H8, H12
# H_n(x) = (-1)^n e^{x^2/2} d^n/dx^n e^{-x^2/2}
# Explicit forms for even n:
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 280x^4 - 840x^2 + 105
# H12 = x^12 - 84x^10 + 1680x^8 - 16800x^6 + 60480x^4 - 30240x^2 + 10395

# Coefficients of q(x) = g(x)/x^2 where g = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(0)=0 is enforced by c3.
# q(x) is a degree 10 polynomial.
# We want to minimize the largest real root r of q(x) where q changes sign.
# value = r^2 / (2*pi)

# Hermite values at 0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1344.0
H12_0 = 209088.0

def _compute_value(c0, c1, c2):
    """Compute the objective value for given coefficients."""
    try:
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Coefficients of q(x) (degree 10 down to 0)
        # q(x) = c3*x^10 - 84*c3*x^8 + (1680*c3 + c2)*x^6 + (-16800*c3 - 28*c2)*x^4 + (60480*c3 + 280*c2 + c1)*x^2 + (-30240*c3 - 840*c2 - 6*c1)
        
        # Note: The constant term of q(x) is -30240*c3 - 840*c2 - 6*c1.
        # For q to divide g exactly by x^2, the constant term of g must be 0, which is satisfied by c3 definition.
        # However, the "divides exactly" condition in the prompt implies g(x) is divisible by x^2.
        # g(x) = x^2 * q(x).
        # The coefficient of x^0 in g is 0 by construction of c3.
        # The coefficient of x^1 in g is always 0 for even Hermite polynomials.
        # So q(x) is well-defined.
        
        # Check if the constant term of q is close to 0? 
        # Actually, the prompt says "Let q = g / x^2, which must divide exactly."
        # This just means g(0)=0 and g'(0)=0.
        # g'(0) is always 0 for even functions.
        # So we just need to find the roots of q.
        
        q_coeffs = np.array([
            c3,          # x^10
            0.0,         # x^9
            -84*c3,      # x^8
            0.0,         # x^7
            1680*c3 + c2, # x^6
            0.0,         # x^5
            -16800*c3 - 28*c2, # x^4
            0.0,         # x^3
            60480*c3 + 280*c2 + c1, # x^2
            0.0,         # x^1
            -30240*c3 - 840*c2 - 6*c1 # x^0
        ])
        
        # Find roots of q(x)
        # Using np.roots can be unstable for high degree, but degree 10 is manageable.
        # Alternative: use companion matrix or specialized solvers.
        try:
            roots = np.roots(q_coeffs)
        except Exception:
            return float('inf')
        
        # Filter real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-9:
                real_roots.append(r.real)
        
        if not real_roots:
            return float('inf')
        
        # For each real root, check if q changes sign (odd multiplicity)
        # We need the LARGEST real root where q changes sign.
        best_r = None
        for r in real_roots:
            # Evaluate q at r - eps and r + eps
            # Use a small epsilon relative to r
            eps = 1e-7 * max(1.0, abs(r))
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            
            # Check sign change
            if q_left * q_right < 0:
                if best_r is None or r > best_r:
                    best_r = r
        
        if best_r is None:
            return float('inf')
            
        # The value is r^2 / (2*pi)
        return (best_r ** 2) / (2 * np.pi)
        
    except Exception:
        return float('inf')

def _objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    return _compute_value(c0, c1, c2)

def construct():
    """Construct three coefficients that minimize the objective value."""
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit
    
    # Seed values
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed.copy()
    best_value = _compute_value(*seed)
    
    # Global Optimization using Differential Evolution
    # Bounds for c0, c1, c2
    # Based on seed, we expect similar magnitudes.
    # c0 ~ -1, c1 ~ 0.04, c2 ~ 0.0003
    bounds = [
        (-10, 10),   # c0
        (-1, 1),     # c1
        (-0.1, 0.1)  # c2
    ]
    
    # Run DE with a reasonable number of iterations
    # maxiter is the number of generations. popsize is the population size.
    # We need to be careful with the time budget.
    # DE can be slow. Let's try a moderate setting first.
    
    if time.time() < deadline - 60:
        try:
            result = differential_evolution(
                _objective, 
                bounds, 
                maxiter=50, 
                popsize=15,
                seed=42,
                tol=1e-10,
                polish=True # Polish with L-BFGS-B at the end
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass

    # If we have time, try a second DE run with different parameters or tighter bounds
    if time.time() < deadline - 30:
        # Try tighter bounds around the current best
        bounds_tight = [
            (best_params[0] - 2, best_params[0] + 2),
            (best_params[1] - 0.5, best_params[1] + 0.5),
            (best_params[2] - 0.05, best_params[2] + 0.05)
        ]
        try:
            result2 = differential_evolution(
                _objective, 
                bounds_tight, 
                maxiter=30, 
                popsize=10,
                seed=123,
                tol=1e-12,
                polish=True
            )
            if result2.fun < best_value:
                best_params = result2.x
                best_value = result2.fun
        except Exception:
            pass
            
    return np.array(best_params)
# EVOLVE-BLOCK-END
