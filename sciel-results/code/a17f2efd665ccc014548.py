"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_value = 2.0
    
    def evaluate(h):
        h = np.array(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if s == 0:
            return float('inf')
        c = np.convolve(h, h)
        max_c = np.max(c)
        val = abs(2 * N * max_c / (s ** 2))
        return val

    # Define a parameterized family: h_i = A * cos(2*pi*f*i/N + phi) + B
    # Or simpler: h_i = 1 + amp * cos(2*pi*f*i/N + phi)
    # We want to keep sum non-zero.
    
    def objective(params):
        amp, f, phi, offset = params
        i = np.arange(N)
        h = offset + amp * np.cos(2 * np.pi * f * i / N + phi)
        return evaluate(h)

    # Bounds for parameters
    # offset: 0.5 to 2.0 (keep positive to avoid sum=0 issues initially)
    # amp: 0.0 to 1.0
    # f: 0.5 to 10.0
    # phi: 0.0 to 2*pi
    bounds = [(0.5, 2.0), (0.0, 1.0), (0.5, 10.0), (0.0, 2 * np.pi)]
    
    # Run differential evolution with limited time
    # maxiter and popsize control the effort
    # We check time in the callback if possible, or just run a fixed small number of iterations
    # DE is slow. Let's use a smaller popsize and maxiter.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=0.01,
            seed=42
        )
        
        if res.fun < best_value:
            amp, f, phi, offset = res.x
            i = np.arange(N)
            best_h = offset + amp * np.cos(2 * np.pi * f * i / N + phi)
            best_value = res.fun
            
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_h

    # Local refinement with Nelder-Mead on the best found
    if best_value < 2.0:
        try:
            from scipy.optimize import minimize
            res_nm = minimize(
                objective, 
                res.x, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if res_nm.fun < best_value:
                amp, f, phi, offset = res_nm.x
                i = np.arange(N)
                best_h = offset + amp * np.cos(2 * np.pi * f * i / N + phi)
                best_value = res_nm.fun
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
