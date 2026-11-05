# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400

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

    # Chirp family: h_i = B + A * cos(2*pi*f0*i/N + pi*k*(i/N)^2 + phi)
    # Using normalized time t = i/N
    def objective(params):
        A, B, f0, k, phi = params
        i = np.arange(N)
        t = i / N
        h = B + A * np.cos(2 * np.pi * f0 * t + np.pi * k * t**2 + phi)
        return evaluate(h)

    # Bounds
    # A: amplitude
    # B: offset
    # f0: initial frequency
    # k: chirp rate
    # phi: phase
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (0.0, 10.0), (-10.0, 10.0), (0.0, 2 * np.pi)]
    
    # Ensure B is not too close to 0 to avoid division by zero issues in optimization landscape
    # DE handles this, but we can restart if it fails.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=0.01,
            seed=42,
            polish=True # Use Nelder-Mead for polish
        )
        
        if res.fun < best_value:
            A, B, f0, k, phi = res.x
            i = np.arange(N)
            t = i / N
            best_h = B + A * np.cos(2 * np.pi * f0 * t + np.pi * k * t**2 + phi)
            best_value = res.fun
            
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_h

    # Local refinement with Nelder-Mead on the best found
    if best_value < 2.0:
        try:
            # Re-run with tighter tolerance if time permits
            # The DE already did polish, but we can do another pass
            # We need to reconstruct the initial guess from res.x
            # res.x is available if res was successful
            if 'res' in locals() and res.success:
                res_nm = minimize(
                    objective, 
                    res.x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                if res_nm.fun < best_value:
                    A, B, f0, k, phi = res_nm.x
                    i = np.arange(N)
                    t = i / N
                    best_h = B + A * np.cos(2 * np.pi * f0 * t + np.pi * k * t**2 + phi)
                    best_value = res_nm.fun
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
