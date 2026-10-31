import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize
from scipy.integrate import simpson
import time

def construct():
    start_time = time.time()
    deadline = start_time + 170  # Safety margin
    
    # Grid setup
    # Use a finer grid for better accuracy of the L1 norm (Simpson's rule)
    # e^{-x^2/2} is negligible beyond |x| > 10-12
    x = np.linspace(-12, 12, 50000)
    
    # Precompute Hermite functions psi_0, psi_4, psi_8, psi_12
    def get_psi(n):
        # psi_n(x) = (2^n n! sqrt(pi))^{-1/2} e^{-x^2/2} H_n(x)
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
        
        # Calculate L2 norm squared using orthonormality: sum of squares of coefficients
        # This is exact and faster than numerical integration
        l2_sq = c0**2 + c4**2 + c8**2 + c12**2
        
        if l2_sq < 1e-15:
            return 1e10
            
        # Calculate L1 norm using Simpson's rule for better accuracy
        # scipy.integrate.simpson requires y and x (or dx)
        # Note: simpson in recent scipy versions uses 'x' keyword or positional
        try:
            l1 = simpson(np.abs(g), x=x)
        except TypeError:
            # Fallback for older scipy if simpson signature differs
            dx = x[1] - x[0]
            l1 = simpson(np.abs(g), dx=dx)
            
        # We want to minimize (L1/L2)^2
        return (l1 / np.sqrt(l2_sq))**2

    # Initial guesses
    # The seed program used [-1.13, 0.04, 0.0003]
    x0_seed = np.array([-1.13, 0.04, 0.0003])
    
    # Generate diverse initial guesses to avoid local minima
    np.random.seed(42)
    initial_guesses = [x0_seed]
    
    # Add random perturbations and random starts
    for _ in range(15):
        # Scale based on the seed magnitude
        c0 = np.random.uniform(-2.0, 0.0)  # Likely negative based on seed
        c4 = np.random.uniform(-0.5, 0.5)
        c8 = np.random.uniform(-0.1, 0.1)
        initial_guesses.append(np.array([c0, c4, c8]))
        
    best_score = float('inf')
    best_params = x0_seed
    
    # Optimize from each initial guess
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 20:
            break
            
        try:
            # Use L-BFGS-B for speed and robustness
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Polishing step with Nelder-Mead if time remains
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
            
    return np.array(best_params)
