"""Improved program: multi-start Nelder-Mead with varied initializations and tighter polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin (evaluator kills at 210s)
    
    # Hermite polynomial values at 0 (physicists' convention)
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 1039500
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Variance (effective n) for each Hermite mode: <x^2> = n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15
            
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        return var_x * var_x

    def objective_precise(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15
            
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        return var_x * var_x

    # Seed coefficients
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_coeffs = seed_coeffs.copy()
    best_score = objective(best_coeffs)
    
    try:
        from scipy.optimize import minimize
        
        # Build a diverse set of initial points
        rng = np.random.RandomState(42)
        initial_points = [seed_coeffs.copy()]
        
        # Add perturbed versions of the seed
        for scale in [0.05, 0.1, 0.2, 0.5]:
            noise = rng.normal(0, scale, size=3)
            initial_points.append(seed_coeffs + noise)
        
        # Add some structured initial points
        # Try points that emphasize different modes
        structured_points = [
            np.array([0.0, 0.0, 0.0]),           # Pure H12
            np.array([1.0, 0.0, 0.0]),           # Emphasize H0
            np.array([0.0, 1.0, 0.0]),           # Emphasize H4
            np.array([0.0, 0.0, 1.0]),           # Emphasize H8
            np.array([-0.5, 0.1, 0.01]),         # Near seed but shifted
            np.array([-1.2, 0.05, 0.001]),       # More negative c0
            np.array([-1.0, 0.04, 0.003]),       # Between seed and pure H0
            np.array([-0.8, 0.06, 0.005]),       # Less negative c0
        ]
        initial_points.extend(structured_points)
        
        # Shuffle to diversify the search order
        rng.shuffle(initial_points)
        
        max_restarts = 12
        restarts_done = 0
        
        for i, x0 in enumerate(initial_points[:max_restarts]):
            if time.time() > deadline - 15:
                break
            
            # Use different tolerances for different restarts
            if i < 3:
                opts = {'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
            else:
                opts = {'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead', options=opts)
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x.copy()
                    restarts_done += 1
                    
                    # Early exit if we're very close to the theoretical minimum
                    if best_score <= 0.25000001:
                        break
                        
            except Exception:
                continue
        
        # Final high-precision polish pass
        if time.time() < deadline - 5:
            try:
                polish_result = minimize(
                    objective_precise,
                    best_coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 100000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                
                if polish_result.fun < best_score:
                    best_score = polish_result.fun
                    best_coeffs = polish_result.x.copy()
                    
            except Exception:
                pass
        
        # One more ultra-fine polish if time allows
        if time.time() < deadline - 2:
            try:
                ultra_result = minimize(
                    objective_precise,
                    best_coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-17, 'adaptive': True}
                )
                
                if ultra_result.fun < best_score:
                    best_score = ultra_result.fun
                    best_coeffs = ultra_result.x.copy()
                    
            except Exception:
                pass
                
    except Exception:
        pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
