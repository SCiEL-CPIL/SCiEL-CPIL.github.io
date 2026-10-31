import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import simpson
import time

def construct():
    start_time = time.time()
    deadline = start_time + 165  # Safety margin (210s limit)
    
    # Grid setup - fine enough for accurate L1 norm
    x = np.linspace(-14, 14, 40000)
    
    # Precompute Hermite functions psi_0, psi_4, psi_8, psi_12
    def get_psi(n):
        norm = np.sqrt(2.0**n * float(factorial(n, exact=True)) * np.sqrt(np.pi))
        H_n = eval_hermite(n, x)
        return (1.0 / norm) * np.exp(-x**2 / 2.0) * H_n
        
    psi_0 = get_psi(0)
    psi_4 = get_psi(4)
    psi_8 = get_psi(8)
    psi_12 = get_psi(12)
    
    # Values at 0 for constraint g(0) = 0
    idx_0 = np.argmin(np.abs(x))
    p0_0 = psi_0[idx_0]
    p0_4 = psi_4[idx_0]
    p0_8 = psi_8[idx_0]
    p0_12 = psi_12[idx_0]
    
    def objective_angles(params):
        theta, phi = params
        # Spherical coordinates for c0, c4, c8 on unit sphere
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        
        # Calculate c12 such that g(0) = 0
        c12 = -(c0 * p0_0 + c4 * p0_4 + c8 * p0_8) / p0_12
        
        g = c0 * psi_0 + c4 * psi_4 + c8 * psi_8 + c12 * psi_12
        
        # L2 norm squared
        l2_sq = 1.0 + c12**2
        
        if l2_sq < 1e-15:
            return 1e10
            
        # L1 norm using Simpson's rule
        try:
            l1 = simpson(np.abs(g), x=x)
        except TypeError:
            dx = x[1] - x[0]
            l1 = simpson(np.abs(g), dx=dx)
            
        # Minimize (L1/L2)^2
        return (l1 / np.sqrt(l2_sq))**2

    # Use Differential Evolution for global optimization on the bounded 2D domain
    bounds = [(0, np.pi), (-np.pi, np.pi)]
    
    best_result = None
    best_score = float('inf')
    
    try:
        res = differential_evolution(objective_angles, bounds, maxiter=100, popsize=20, tol=1e-8, seed=42)
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
        
    # Polish with Nelder-Mead if time remains
    if time.time() < deadline - 10:
        if best_result is None:
            best_result = np.array([np.pi/4, 0.0])
            
        try:
            res_nm = minimize(objective_angles, best_result, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_result = res_nm.x
        except Exception:
            pass
            
    # Convert back to coefficients c0, c4, c8
    theta, phi = best_result
    c0 = np.sin(theta) * np.cos(phi)
    c4 = np.sin(theta) * np.sin(phi)
    c8 = np.cos(theta)
    
    return np.array([c0, c4, c8])
