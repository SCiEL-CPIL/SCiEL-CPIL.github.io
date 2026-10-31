import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize
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
    
    def objective(params):
        c0, c4, c8 = params
        # Calculate c12 such that g(0) = 0
        c12 = -(c0 * p0_0 + c4 * p0_4 + c8 * p0_8) / p0_12
        
        g = c0 * psi_0 + c4 * psi_4 + c8 * psi_8 + c12 * psi_12
        
        # L2 norm squared using orthonormality
        l2_sq = c0**2 + c4**2 + c8**2 + c12**2
        
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

    # Use a multi-start approach with diverse initial guesses
    np.random.seed(12345)
    
    # The seed program's initial guess
    x0_seed = np.array([-1.13, 0.04, 0.0003])
    
    # Generate diverse initial guesses
    initial_guesses = [x0_seed]
    
    # Random perturbations around seed
    for _ in range(8):
        c0 = x0_seed[0] + np.random.uniform(-0.5, 0.5)
        c4 = x0_seed[1] + np.random.uniform(-0.3, 0.3)
        c8 = x0_seed[2] + np.random.uniform(-0.05, 0.05)
        initial_guesses.append(np.array([c0, c4, c8]))
    
    # Random starts in reasonable ranges
    for _ in range(12):
        c0 = np.random.uniform(-3.0, 1.0)
        c4 = np.random.uniform(-1.0, 1.0)
        c8 = np.random.uniform(-0.5, 0.5)
        initial_guesses.append(np.array([c0, c4, c8]))
    
    best_score = float('inf')
    best_params = x0_seed.copy()
    
    # Optimize from each initial guess
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 30:
            break
            
        try:
            # Use L-BFGS-B for speed and robustness
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polishing step with Nelder-Mead if time remains
    if time.time() < deadline - 15:
        try:
            res_nm = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
