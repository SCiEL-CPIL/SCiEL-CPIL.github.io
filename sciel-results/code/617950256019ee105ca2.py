import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin
    
    # Grid setup
    # High resolution grid for accurate L1 and L2 norms
    # e^{-x^2/2} is negligible beyond |x| > 12
    x = np.linspace(-12, 12, 20000)
    dx = x[1] - x[0]
    
    # Precompute Hermite functions psi_0, psi_4, psi_8, psi_12
    def get_psi(n):
        # psi_n(x) = (2^n n! sqrt(pi))^{-1/2} e^{-x^2/2} H_n(x)
        # Using scipy.special.eval_hermite for H_n
        # Normalization factor
        norm = np.sqrt(2.0**n * float(factorial(n, exact=True)) * np.sqrt(np.pi))
        H_n = eval_hermite(n, x)
        return (1.0 / norm) * np.exp(-x**2 / 2.0) * H_n
        
    psi_0 = get_psi(0)
    psi_4 = get_psi(4)
    psi_8 = get_psi(8)
    psi_12 = get_psi(12)
    
    # Values at 0 for constraint g(0) = 0
    # We can compute these analytically or from the grid.
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    # norm_0 = sqrt(1 * 1 * sqrt(pi)) = pi^0.25
    # norm_4 = sqrt(16 * 24 * sqrt(pi)) = sqrt(384 * sqrt(pi))
    # norm_8 = sqrt(256 * 40320 * sqrt(pi))
    # norm_12 = sqrt(4096 * 479001600 * sqrt(pi))
    
    # Using grid values is safer to avoid manual calculation errors, 
    # but finding index closest to 0 is O(N) or O(log N).
    idx_0 = np.argmin(np.abs(x))
    p0_0 = psi_0[idx_0]
    p0_4 = psi_4[idx_0]
    p0_8 = psi_8[idx_0]
    p0_12 = psi_12[idx_0]
    
    def objective(params):
        c0, c4, c8 = params
        # Calculate c12 such that g(0) = 0
        # c0*p0_0 + c4*p0_4 + c8*p0_8 + c12*p0_12 = 0
        c12 = -(c0 * p0_0 + c4 * p0_4 + c8 * p0_8) / p0_12
        
        g = c0 * psi_0 + c4 * psi_4 + c8 * psi_8 + c12 * psi_12
        
        # Calculate L2 norm squared and L1 norm
        l2_sq = np.sum(g**2) * dx
        if l2_sq < 1e-10:
            return 1e10
        l1 = np.sum(np.abs(g)) * dx
        
        # We want to minimize (L1/L2)^2
        return (l1 / np.sqrt(l2_sq))**2

    # Initial guesses
    # Seed guess
    x0_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Generate multiple random initial guesses
    # The coefficients are likely of similar magnitude or smaller
    # Let's sample from a normal distribution
    np.random.seed(42)
    initial_guesses = [x0_seed]
    for _ in range(10):
        # Random start in a reasonable box
        c0 = np.random.uniform(-2, 2)
        c4 = np.random.uniform(-0.5, 0.5)
        c8 = np.random.uniform(-0.1, 0.1)
        initial_guesses.append(np.array([c0, c4, c8]))
        
    best_score = float('inf')
    best_params = x0_seed
    
    # Optimize from each initial guess
    for x0 in initial_guesses:
        if time.time() > deadline - 20:
            break
            
        try:
            # Use L-BFGS-B for speed and robustness
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
        except Exception:
            pass
            
    # If time remains, try Powell for the best found so far to polish
    if time.time() < deadline - 30:
        try:
            res_powell = minimize(objective, best_params, method='Powell', options={'maxiter': 1000, 'ftol': 1e-14})
            if res_powell.fun < best_score:
                best_score = res_powell.fun
                best_params = res_powell.x
        except Exception:
            pass

    return np.array(best_params)
