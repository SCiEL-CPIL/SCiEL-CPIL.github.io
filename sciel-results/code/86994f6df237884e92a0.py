import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# Precompute grid and weights
# x range [-6, 6] is sufficient for e^{-pi x^2}
x = np.linspace(-6, 6, 20001)
dx = x[1] - x[0]
w1 = np.exp(-np.pi * x**2)
w2 = np.exp(-2.0 * np.pi * x**2)

# Hermite polynomials
def H0(x): return np.ones_like(x)
def H4(x): return x**4 - 6*x**2 + 3
def H8(x): return x**8 - 24*x**6 + 72*x**4 - 32*x**2 + 15
def H12(x): return x**12 - 60*x**10 + 300*x**8 - 1200*x**6 + 1200*x**4 - 168*x**2 + 105

# Precompute polynomial basis on grid
H0_g = H0(x)
H4_g = H4(x)
H8_g = H8(x)
H12_g = H12(x)

def objective(c):
    """
    Minimize (A(f) * A(F(f))) / ||f||_2^2
    which is (I^2) / Q
    I = integral |P(x)| e^{-pi x^2} dx
    Q = integral P(x)^2 e^{-2pi x^2} dx
    """
    c0, c4, c8 = c
    
    # Avoid division by zero if c is all zeros
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        return 1e100
    
    # c12 from P(0) = 0
    # P(0) = c0*1 + c4*3 + c8*15 + c12*105 = 0
    c12 = -(c0 + 3*c4 + 15*c8) / 105.0
    
    # Compute P(x)
    P = c0 * H0_g + c4 * H4_g + c8 * H8_g + c12 * H12_g
    
    # Compute integrals
    abs_P = np.abs(P)
    I = np.trapz(abs_P * w1, x)
    Q = np.trapz(P**2 * w2, x)
    
    if Q < 1e-12:
        return 1e100
        
    return (I * I) / Q

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # 1. Global Optimization with Differential Evolution
    # Bounds for c0, c4, c8
    # Since the objective is scale invariant, the specific magnitude doesn't matter,
    # but DE needs a bounded domain.
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8,
            polish=False # We will polish manually
        )
        best_c = res_de.x
        best_val = res_de.fun
    except Exception:
        # Fallback to seed if DE fails
        best_c = np.array([-1.13, 0.04, 0.0003])
        best_val = objective(best_c)
        
    # Check time
    if time.time() > deadline - 30:
        return best_c

    # 2. Local Refinement with Powell (robust to non-smoothness)
    try:
        res_powell = minimize(
            objective, 
            best_c, 
            method='Powell', 
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res_powell.fun < best_val:
            best_val = res_powell.fun
            best_c = res_powell.x
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 10:
        return best_c

    # 3. Final Polish with Nelder-Mead (often better for final tweaks)
    try:
        res_nm = minimize(
            objective, 
            best_c, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_c = res_nm.x
    except Exception:
        pass

    return best_c
