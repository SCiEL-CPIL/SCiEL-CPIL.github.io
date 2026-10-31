import numpy as np
import scipy.optimize as opt
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds buffer
    
    # Objective function to minimize: -Ratio
    def neg_ratio(h):
        h = np.asarray(h)
        # Handle potential zeros or very small values
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution
        # h*h has length 2*N - 1
        conv = np.convolve(h, h, mode='full')
        l_inf = np.max(conv)
        if l_inf < 1e-12:
            return 0.0
        ratio = l2_sq / (l1 * l_inf)
        return -ratio

    # Initial guess: uniform
    x0 = np.ones(N)
    
    # 1. Global Search with Differential Evolution
    # Bounds: [0, 1] is sufficient due to scale invariance
    bounds = [(0.0, 1.0)] * N
    
    best_x = x0
    best_val = neg_ratio(x0)
    
    try:
        # DE is stochastic, use a seed for reproducibility
        res_de = opt.differential_evolution(
            neg_ratio, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-8,
            polish=False # We will polish manually
        )
        
        if res_de.success and res_de.fun < best_val:
            best_x = res_de.x
            best_val = res_de.fun
            
        # Check time
        if time.time() > deadline - 20:
            return list(best_x)
            
    except Exception:
        pass

    # 2. Local Refinement with L-BFGS-B
    # We can try to refine the DE result
    try:
        res_lbfgs = opt.minimize(
            neg_ratio, 
            best_x, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res_lbfgs.success and res_lbfgs.fun < best_val:
            best_x = res_lbfgs.x
            best_val = res_lbfgs.fun
            
    except Exception:
        pass

    # 3. Final Polish with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            res_polish = opt.minimize(
                neg_ratio, 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_polish.success and res_polish.fun < best_val:
                best_x = res_polish.x
                best_val = res_polish.fun
        except Exception:
            pass

    return list(best_x)
