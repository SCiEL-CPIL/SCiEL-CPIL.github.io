"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

# Pre-compute Hermite polynomial coefficients for x^0, x^2, ..., x^12
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 138600x^4 - 332640x^2 + 10395

# Coefficients for g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(x) = g0 + g2*x^2 + ... + g12*x^12
# g0 = c0*1 + c1*3 + c2*105 + c3*10395 = 0 => c3 = -(c0 + 3c1 + 105c2) / 10395
# g2 = c1*(-6) + c2*(-420) + c3*(-332640)
# g4 = c1*(1) + c2*(210) + c3*(138600)
# g6 = c2*(-28) + c3*(-25200)
# g8 = c2*(1) + c3*(2100)
# g10 = c3*(-84)
# g12 = c3*(1)

# Q(y) = g12*y^5 + g10*y^4 + g8*y^3 + g6*y^2 + g4*y + g2
# We want the largest positive real root R of Q(y) where sign changes.
# Value = R / (2 * pi)

def compute_value(c0, c1, c2):
    # Calculate c3
    c3 = -(c0 + 3.0*c1 + 105.0*c2) / 10395.0
    
    # Calculate coefficients of Q(y)
    # Order: [g12, g10, g8, g6, g4, g2]
    g12 = c3
    g10 = -84.0 * c3
    g8 = 2100.0 * c3
    g6 = -25200.0 * c3 - 28.0 * c2
    g4 = 138600.0 * c3 + 210.0 * c2 + c1
    g2 = -332640.0 * c3 - 420.0 * c2 - 6.0 * c1
    
    coeffs = np.array([g12, g10, g8, g6, g4, g2])
    
    # Find roots
    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e10

    # Filter for real, positive roots
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9:
            r_val = root.real
            if r_val > 1e-9:
                # Check sign change
                # Evaluate Q(y) at r_val +/- eps
                eps = 1e-6
                y1 = r_val - eps
                y2 = r_val + eps
                # Use Horner's method or polyval
                q1 = np.polyval(coeffs, y1)
                q2 = np.polyval(coeffs, y2)
                if q1 * q2 < 0:
                    valid_roots.append(r_val)
                    
    if not valid_roots:
        return 1e10
        
    R_max = max(valid_roots)
    value = R_max / (2.0 * np.pi)
    return value

def construct():
    # Global optimization using Differential Evolution
    # This is a different approach family (Global Stochastic) vs Local (Nelder-Mead)
    
    # Define bounds
    # Seed is [-1.13, 0.04, 0.0003]
    # c2 is small, c1 is small, c0 is larger.
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-1.0, 1.0)]
    
    # Objective function
    def objective(coeffs):
        return compute_value(coeffs[0], coeffs[1], coeffs[2])
        
    # Run differential evolution
    # popsize=15, maxiter=100, tol=1e-8
    # We need to be careful with time. 180s limit.
    # Let's run it with a timeout wrapper or just conservative settings.
    
    start_time = time.time()
    timeout = 150.0 # Leave 30s buffer
    
    def timed_objective(coeffs):
        if time.time() - start_time > timeout:
            return 1e10
        return objective(coeffs)
        
    try:
        result = differential_evolution(
            timed_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-8,
            polish=True
        )
        
        if result.fun < 1e9: # Valid result
            # Polish with Nelder-Mead to squeeze out more? 
            # The directive says "do not reuse current designs", but using DE as primary is the new design.
            # Adding a local polish step is standard practice, but let's stick to DE result to be safe on "approach family".
            # Actually, DE often ends near a good local min.
            return result.x
            
    except Exception:
        pass
        
    # Fallback to seed if DE fails or times out
    return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
# EVOLVE-BLOCK-END
