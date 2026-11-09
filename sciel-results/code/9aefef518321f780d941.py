"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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

    # Effective quantum numbers (n + 1/2)
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
        # Product of variances (isotropic symmetry implies var_xi = var_x)
        return (var_x * var_x)

    # Theoretical analysis suggests the global minimum is at c4=0, c8=0
    # because n0 < n4 < n8 < n12, and the constraint on c12 introduces
    # negligible weight for large H12(0).
    # We start at this analytical solution.
    best_coeffs = np.array([1.0, 0.0, 0.0])
    best_score = objective(best_coeffs)
    
    # Perform a high-precision local polish to ensure we are at the exact minimum
    # in case of any subtle numerical deviations or if my analytical assumption
    # about the cross-terms or constraint impact is slightly off.
    try:
        from scipy.optimize import minimize
        
        # Polish around the analytical solution
        if time.time() < deadline - 10:
            # Use a very tight tolerance to confirm the minimum
            result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
                
            # If the polish moved us significantly away from (1,0,0), 
            # it might indicate a better spot, but given the structure, 
            # (1,0,0) is extremely robust.
            
            # Also check a few other points just in case, but expect no improvement
            test_points = [
                np.array([1.0, 0.0, 0.0]),
                np.array([1.0, 1e-6, 0.0]),
                np.array([1.0, 0.0, 1e-6]),
                np.array([1.0, -1e-6, 0.0]),
                np.array([1.0, 0.0, -1e-6]),
            ]
            
            for tp in test_points:
                sc = objective(tp)
                if sc < best_score:
                    best_score = sc
                    best_coeffs = tp.copy()
                    
                    # If we found a better point, polish it
                    if time.time() < deadline - 5:
                        res2 = minimize(objective, best_coeffs, method='Nelder-Mead', 
                                        options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17})
                        if res2.fun < best_score:
                            best_score = res2.fun
                            best_coeffs = res2.x.copy()

    except Exception:
        pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
