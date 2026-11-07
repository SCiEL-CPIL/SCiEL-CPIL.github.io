"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# Coefficients for Hermite polynomials H_0, H_4, H_8, H_12
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 60x^10 + 1500x^8 - 15400x^6 + 63000x^4 - 84000x^2 + 10395

# We express g(x) = x^2 * q(x).
# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# Since g(0)=0, the constant term is 0.
# Let y = x^2.
# H0 = 1
# H4 = y^2 - 6y + 3
# H8 = y^4 - 28y^3 + 210y^2 - 420y + 105
# H12 = y^6 - 60y^5 + 1500y^4 - 15400y^3 + 63000y^2 - 84000y + 10395

# g(x) = c0(1) + c1(y^2 - 6y + 3) + c2(y^4 - 28y^3 + 210y^2 - 420y + 105) + c3(y^6 - 60y^5 + 1500y^4 - 15400y^3 + 63000y^2 - 84000y + 10395)
# g(0) corresponds to y=0.
# g(0) = c0 + 3c1 + 105c2 + 10395c3 = 0
# c3 = -(c0 + 3c1 + 105c2) / 10395

# q(y) = g(x)/x^2 = g(sqrt(y))/y
# The polynomial P(y) representing g(x) in terms of y is:
# P(y) = c0*1 + c1*(y^2 - 6y + 3) + c2*(y^4 - 28y^3 + 210y^2 - 420y + 105) + c3*(y^6 - 60y^5 + 1500y^4 - 15400y^3 + 63000y^2 - 84000y + 10395)
# Note that P(0) = 0.
# So P(y) = y * Q(y).
# Q(y) is a polynomial of degree 5.
# We need the largest real positive root of Q(y). Let this root be y_r.
# Then r = sqrt(y_r).
# Value = r^2 / (2*pi) = y_r / (2*pi).

def compute_q_coeffs(c0, c1, c2):
    """
    Returns the coefficients of Q(y) such that g(x) = x^2 * Q(x^2).
    Q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    """
    c3 = -(c0 + 3*c1 + 105*c2) / 10395.0
    
    # Coefficients of P(y) (which is g in terms of y)
    # P(y) = c3*y^6 + (c2 - 60*c3)*y^5 + (210*c2 - 28*c1 + 1500*c3)*y^4 + ...
    
    # Let's build P(y) coefficients from degree 6 down to 0
    # P(y) = sum_{k=0}^6 p_k y^k
    
    # H0: 1
    # H4: 3 - 6y + y^2
    # H8: 105 - 420y + 210y^2 - 28y^3 + y^4
    # H12: 10395 - 84000y + 63000y^2 - 15400y^3 + 1500y^4 - 60y^5 + y^6
    
    p = np.zeros(7)
    
    # p[0] (const)
    p[0] = c0 + 3*c1 + 105*c2 + 10395*c3 # Should be 0
    
    # p[1] (y^1)
    p[1] = -6*c1 - 420*c2 - 84000*c3
    
    # p[2] (y^2)
    p[2] = c1 + 210*c2 + 63000*c3
    
    # p[3] (y^3)
    p[3] = -28*c2 - 15400*c3
    
    # p[4] (y^4)
    p[4] = c2 + 1500*c3
    
    # p[5] (y^5)
    p[5] = -60*c3
    
    # p[6] (y^6)
    p[6] = c3
    
    # Q(y) = P(y) / y
    # So q_coeffs for Q(y) of degree 5 are p[1] to p[6]
    # Q(y) = q0 + q1*y + q2*y^2 + q3*y^3 + q4*y^4 + q5*y^5
    q_coeffs = p[1:7]
    
    return q_coeffs

def objective(c):
    c0, c1, c2 = c
    q_coeffs = compute_q_coeffs(c0, c1, c2)
    
    # Find roots of Q(y)
    # numpy.roots expects coefficients from highest degree to lowest
    roots = np.roots(q_coeffs[::-1])
    
    # Filter real, positive roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    pos_roots = real_roots[real_roots > 1e-9]
    
    if len(pos_roots) == 0:
        # If no positive real roots, the "largest root" logic is tricky.
        # If Q(y) has no positive roots, does it mean r is 0?
        # The problem states "r be the largest real root of q at which q changes sign".
        # If no such root exists, the value might be considered 0 or invalid.
        # However, for a valid inequality bound, we likely need a positive root.
        # If Q(y) is always positive for y>0, then g(x)/x^2 > 0 for x!=0.
        # In that case, the "largest root" doesn't exist.
        # But usually, these polynomials will have roots.
        # If no roots, return a large penalty to discourage this.
        return 1e9
        
    max_root = np.max(pos_roots)
    
    # Value is max_root / (2*pi)
    return max_root / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin before 210s kill
    
    # Seed
    seed_c = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_c = seed_c.copy()
    best_val = objective(best_c)
    
    # Differential Evolution
    # Bounds for c0, c1, c2
    # The seed is around [-1.13, 0.04, 0.0003]
    # Let's explore a reasonable range.
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Check time
    if time.time() - start_time > deadline - 30:
        return best_c
    
    try:
        # Use differential evolution for global search
        # maxiter is controlled by time, but we can set a high limit and check time in callback
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True, # Use L-BFGS-B to polish the result
            init='sobol'
        )
        
        if result.fun < best_val:
            best_c = result.x
            best_val = result.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Check time for local refinement
    if time.time() - start_time < deadline - 10:
        try:
            # Refine with Nelder-Mead or L-BFGS-B
            # L-BFGS-B requires gradient, which we don't have analytically easily.
            # Nelder-Mead is derivative-free and robust.
            result_local = minimize(
                objective,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result_local.fun < best_val:
                best_c = result_local.x
                best_val = result_local.fun
                
        except Exception as e:
            print(f"Local minimization failed: {e}")
            
    return best_c

# EVOLVE-BLOCK-END
