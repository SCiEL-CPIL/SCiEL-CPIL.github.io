"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 600
DEADLINE = 145.0  # safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute FFT size for efficient convolution
    n_fft = 2 * N
    
    # Compute the true objective: max(autoconv) / sum(x^2)
    def objective(x):
        x = np.maximum(x, 0)
        # Linear autoconvolution via FFT
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real[:2*N-1]
        norm = np.sum(x * x)
        if norm < 1e-15:
            return 1e15
        return np.max(conv) / norm
    
    # Smooth surrogate using log-sum-exp for the max
    def obj_smooth(x, beta=50.0):
        x = np.maximum(x, 0)
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real[:2*N-1]
        norm = np.sum(x * x)
        if norm < 1e-15:
            return 1e15
        conv_norm = conv / norm
        # Log-sum-exp approximation of max with temperature beta
        m = np.max(conv_norm)
        # Use scaled version for numerical stability
        scaled = beta * (conv_norm - m)
        # Clip for numerical stability
        scaled = np.clip(scaled, -100, 100)
        log_sum = np.log(np.sum(np.exp(scaled))) / beta
        return m + log_sum / beta
    
    # Initial uniform value
    best_x = np.ones(N)
    best_val = objective(best_x)
    
    # Try optimization with L-BFGS-B using smooth surrogate
    try:
        # Diverse initializations
        inits = [
            np.ones(N),
            np.concatenate([np.ones(N//2), np.zeros(N//2)]),
            np.concatenate([np.zeros(N//4), np.ones(N//2), np.zeros(N//4)]),
            np.random.dirichlet(np.ones(N) * 5) * N,
            np.random.dirichlet(np.ones(N) * 20) * N,
        ]
        
        bounds = [(0, None)] * N
        
        for init in inits:
            if time.time() > DEADLINE - 20:
                break
            
            x0 = init.copy()
            x0 = np.maximum(x0, 0)
            
            # First optimize with smooth surrogate
            res = minimize(
                obj_smooth,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            
            if res.fun < best_val:
                x_opt = np.maximum(res.x, 0)
                val = objective(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
            
            # Polish with true objective if time permits
            if time.time() < DEADLINE - 15:
                res2 = minimize(
                    objective,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res2.fun < best_val:
                    x_opt = np.maximum(res2.x, 0)
                    val = objective(x_opt)
                    if val < best_val:
                        best_val = val
                        best_x = x_opt.copy()
    
    except Exception:
        pass
    
    # Final high-precision polish if time remains
    if time.time() < DEADLINE - 5:
        try:
            bounds = [(0, None)] * N
            res3 = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res3.fun < best_val:
                x_opt = np.maximum(res3.x, 0)
                val = objective(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_x = np.maximum(best_x, 0)
    
    return list(best_x)
# EVOLVE-BLOCK-END
