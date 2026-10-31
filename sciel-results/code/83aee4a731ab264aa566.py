"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Indices of the Hermite polynomials
    # c0, c4, c8 are free. c12 is fixed by P(0)=0.
    # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # c0 - 3c4 + 105c8 - 10395*c12 = 0
    # c12 = (c0 - 3c4 + 105c8) / 10395
    
    def objective(x):
        c0, c4, c8 = x
        c12 = (c0 - 3.0*c4 + 105.0*c8) / 10395.0
        c = np.array([c0, c4, c8, c12])
        
        # Calculate A(f)^2 and A(f_hat)^2
        # A(f)^2 = <x^2> = c^T K c / c^T c
        # A(f_hat)^2 = <xi^2> = c^T L c / c^T c
        
        # K and L are diagonal in the subspace {0, 4, 8, 12}
        # K_ii = i + 0.5
        # L_ii = i + 0.5 (due to symmetry in this subspace)
        
        diag = np.array([0.5, 4.5, 8.5, 12.5])
        
        num = np.sum(diag * c**2)
        den = np.sum(c**2)
        
        if den < 1e-12:
            return 1e9
            
        # The product is sqrt(num/den * num/den) = num/den
        # Wait, A(f)A(f_hat) = sqrt(<x^2>) * sqrt(<xi^2>) = sqrt(num/den) * sqrt(num/den) = num/den.
        # So we minimize num/den.
        
        return num / den

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    best_val = objective(x0)
    best_x = x0.copy()
    
    # Multi-start optimization
    # Use a mix of L-BFGS-B and Nelder-Mead for robustness
    starts = [x0]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(50):
        starts.append(np.random.uniform(-2.0, 2.0, 3))
        
    # Add grid points
    grid_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for c0 in grid_vals:
        for c4 in grid_vals:
            for c8 in grid_vals:
                starts.append(np.array([c0, c4, c8]))
                
    for x_start in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
            # Also try Nelder-Mead for local refinement if L-BFGS-B gets stuck
            res_nm = minimize(objective, x_start, method='Nelder-Mead', options={'maxiter': 1000})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
                
        except Exception:
            continue
            
    return best_x

# EVOLVE-BLOCK-END
