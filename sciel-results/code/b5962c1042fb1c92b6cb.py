"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin
    
    # Hermite polynomials H0, H4, H8, H12
    # We need to compute c12 from c0, c4, c8 such that g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 9240
    
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0
    
    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # Precompute Hermite values on a grid
    # Range [-10, 10] is sufficient for Gaussian
    x = np.linspace(-10, 10, 2000)
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    exp_term = np.exp(-x**2 / 2)
    
    def objective(coeffs):
        if time.time() > deadline:
            return 1e10
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Compute g(x)
        g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * exp_term
        
        # Enforce L2 norm = 1
        norm_sq = np.trapz(np.abs(g)**2, x)
        
        if norm_sq < 1e-12:
            return 1e10
            
        # Normalize g to L2 norm 1
        g_norm = g / np.sqrt(norm_sq)
        
        # Calculate L1 norm
        l1_norm = np.trapz(np.abs(g_norm), x)
        
        return l1_norm

    # Initial guess from seed
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    # Use Differential Evolution for robust global search
    # Bounds are arbitrary but should be large enough to cover the optimal region
    # The seed values are O(1), so [-10, 10] is a reasonable start
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_result = None
    best_score = objective([c0_init, c4_init, c8_init])
    
    try:
        # Run Differential Evolution
        de_res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-5,
            popsize=15,
            workers=-1
        )
        
        if de_res.fun < best_score:
            best_score = de_res.fun
            best_result = de_res.x
        else:
            best_result = [c0_init, c4_init, c8_init]
            
    except Exception:
        best_result = [c0_init, c4_init, c8_init]
    
    # Polish with Nelder-Mead if time permits
    if time.time() < deadline - 5:
        try:
            nm_res = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if nm_res.fun < best_score:
                best_score = nm_res.fun
                best_result = nm_res.x
        except Exception:
            pass
    
    return np.array(best_result)

# EVOLVE-BLOCK-END
