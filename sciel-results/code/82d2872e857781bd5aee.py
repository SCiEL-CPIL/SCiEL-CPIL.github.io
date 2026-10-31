import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin
    
    # Uniform baseline
    h_uniform = np.ones(N) / N
    c_uniform = np.convolve(h_uniform, h_uniform, mode='full')
    val_uniform = np.max(c_uniform) / (np.sum(h_uniform)**2)
    
    best_h = h_uniform
    best_val = val_uniform
    
    def objective(h):
        s = np.sum(h)
        if s <= 1e-12:
            return 1e10
        h_norm = h / s
        c = np.convolve(h_norm, h_norm, mode='full')
        return np.max(c)
        
    # Try a few random initializations
    initializations = [
        np.ones(N) / N,
        np.exp(-np.linspace(-1, 1, N)**2 * 5), # Gaussian-ish
        np.sin(np.linspace(0, np.pi, N)), # Sine bump
        np.random.exponential(1.0, N)
    ]
    
    for i in range(len(initializations)):
        init_h = initializations[i]
        init_h = init_h / np.sum(init_h)
        
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init_h, method='L-BFGS-B', 
                           bounds=[(0, None)] * N,
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                val = res.fun
                # Convert back to the comparable scale
                # res.fun is max(c) for normalized h.
                # val_uniform is max(c) for normalized h_uniform.
                if val < best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            pass
            
    # If time remains, do a few more random restarts
    while time.time() < deadline - 5.0:
        init_h = np.random.exponential(1.0, N)
        init_h = init_h / np.sum(init_h)
        try:
            res = minimize(objective, init_h, method='L-BFGS-B', 
                           bounds=[(0, None)] * N,
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            pass
            
    return list(best_h)
