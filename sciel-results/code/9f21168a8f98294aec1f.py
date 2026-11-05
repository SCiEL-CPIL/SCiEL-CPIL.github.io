"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_value = 2.0
    
    def evaluate(h):
        h = np.array(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        c = np.convolve(h, h)
        max_c = np.max(c)
        val = abs(2 * N * max_c / (s ** 2))
        return val

    # Zadoff-Chu inspired: h_i = B + A*Re(exp(1j*(2*pi*f0*i/N + pi*k*(i/N)^2 + phi)))
    # Plus a second harmonic to break symmetry
    def objective(params):
        A, B, f0, k, phi, A2, f2, phi2 = params
        i = np.arange(N)
        t = i / N
        h = B + A * np.cos(2 * np.pi * f0 * t + np.pi * k * t * t + phi)
        h += A2 * np.cos(2 * np.pi * f2 * t + phi2)
        return evaluate(h)

    # Bounds: keep B away from 0 to avoid sum=0
    bounds = [
        (-3.0, 3.0),   # A
        (0.1, 3.0),    # B (keep positive to ensure nonzero sum)
        (0.1, 15.0),   # f0
        (-10.0, 10.0), # k (chirp rate)
        (0.0, 2 * np.pi), # phi
        (-3.0, 3.0),   # A2
        (0.1, 15.0),   # f2
        (0.0, 2 * np.pi)  # phi2
    ]
    
    # Multi-restart differential evolution
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        try:
            seed_val = 42 + restart * 100
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=40, 
                popsize=8, 
                tol=0.005,
                seed=seed_val,
                init='sobol'
            )
            
            if res.fun < best_value:
                A, B, f0, k, phi, A2, f2, phi2 = res.x
                i = np.arange(N)
                t = i / N
                best_h = B + A * np.cos(2 * np.pi * f0 * t + np.pi * k * t * t + phi)
                best_h += A2 * np.cos(2 * np.pi * f2 * t + phi2)
                best_value = res.fun
                
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_h

    # Local refinement with Nelder-Mead on the best found
    if best_value < 2.0:
        try:
            # Reconstruct params from best_h is hard; use last successful res
            # Instead, do a few local perturbations
            i = np.arange(N)
            t = i / N
            
            # Try Nelder-Mead with the last res.x if available
            if 'res' in locals() and res.success:
                res_nm = minimize(
                    objective, 
                    res.x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res_nm.fun < best_value:
                    A, B, f0, k, phi, A2, f2, phi2 = res_nm.x
                    best_h = B + A * np.cos(2 * np.pi * f0 * t + np.pi * k * t * t + phi)
                    best_h += A2 * np.cos(2 * np.pi * f2 * t + phi2)
                    best_value = res_nm.fun
        except Exception:
            pass
    
    # Final time check
    if time.time() > deadline:
        return best_h

    return best_h
# EVOLVE-BLOCK-END
