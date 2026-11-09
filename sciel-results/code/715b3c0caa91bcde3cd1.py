"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 180s limit
    
    # Hermite polynomial values at 0
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 1039500
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Coefficients of Hermite polynomials for x^2 moments (n + 1/2)
    # <x^2> for psi_n is n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Weights (squared coefficients)
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15 # Avoid division by zero, penalize 0 norm
            
        # Variance (effective n)
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        # Assuming isotropic symmetry, var_xi = var_x
        # Product of variances
        return (var_x * var_x)

    # Seed coefficients
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_coeffs = seed_coeffs.copy()
    best_score = objective(best_coeffs)
    
    # Try to optimize using Nelder-Mead with multiple restarts and polish
    try:
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        
        # First, try a broader global search with differential evolution
        # to find a good starting point
        if time.time() < deadline - 60:
            bounds = [(-1, 1), (-1, 1), (-0.1, 0.1)]
            de_result = differential_evolution(
                objective, bounds, 
                maxiter=1000, tol=1e-10, 
                popsize=20, seed=42,
                polish=True
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.copy()
        
        # Then refine with Nelder-Mead from multiple starting points
        if time.time() < deadline - 10:
            # Generate diverse starting points
            start_points = [best_coeffs.copy()]
            
            # Add perturbed versions around the best found
            for scale in [0.01, 0.05, 0.1, 0.5]:
                for _ in range(3):
                    noise = np.random.normal(0, scale, size=3)
                    start_points.append(best_coeffs + noise)
            
            # Also try the original seed
            start_points.append(seed_coeffs.copy())
            
            for x0 in start_points:
                if time.time() > deadline - 15:
                    break
                
                result = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x.copy()
                    
                    # If we found something better, do an extra polish
                    if time.time() < deadline - 5 and best_score > 0.2500001:
                        polish_result = minimize(objective, best_coeffs, method='Nelder-Mead',
                                                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16})
                        if polish_result.fun < best_score:
                            best_score = polish_result.fun
                            best_coeffs = polish_result.x.copy()
                
                # Early exit if we hit the theoretical minimum for Gaussian (0.25)
                if best_score <= 0.2500001:
                    break
                    
    except Exception:
        pass

    # Final high-precision polish if time permits
    if time.time() < deadline - 2 and best_score > 0.2500001:
        try:
            final_polish = minimize(objective, best_coeffs, method='Nelder-Mead',
                                    options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17})
            if final_polish.fun < best_score:
                best_score = final_polish.fun
                best_coeffs = final_polish.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
