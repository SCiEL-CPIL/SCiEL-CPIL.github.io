import numpy as np
from scipy import optimize
import time

N = 600
DELTA = (1.0/2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return np.inf
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac * 0.5 / (integral ** 2)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with uniform
    h = np.ones(N)
    best_score = compute_score(h)
    best_h = h.copy()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N  # midpoints
    
    # 1. Parametric Fourier Optimization
    # Use more modes for flexibility
    n_modes = 10
    def param_to_h(params):
        h = 1.0
        for k in range(1, n_modes + 1):
            h += params[2*k-2] * np.cos(2*np.pi*k*x)
            h += params[2*k-1] * np.sin(2*np.pi*k*x)
        return np.maximum(h, 0)
        
    def objective_fourier(params):
        h = param_to_h(params)
        return compute_score(h)
        
    # Multi-start L-BFGS-B
    np.random.seed(42)
    best_fourier_score = np.inf
    best_fourier_params = None
    
    for restart in range(15):
        if time.time() > deadline - 20:
            break
        if restart == 0:
            x0 = np.zeros(2 * n_modes)
        else:
            x0 = np.random.randn(2 * n_modes) * 0.2
            
        # L-BFGS-B is efficient for smooth objectives
        res = optimize.minimize(objective_fourier, x0, method='L-BFGS-B', 
                                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
        if res.fun < best_fourier_score:
            best_fourier_score = res.fun
            best_fourier_params = res.x
            best_h = param_to_h(res.x).copy()
            best_score = res.fun
            
    # 2. Direct Optimization on 600 heights (Polish)
    # Start from best Fourier result
    if time.time() < deadline - 10:
        h_start = best_h.copy()
        
        # We can use L-BFGS-B on the 600 variables.
        # Finite difference gradient will be expensive but doable for a few iterations.
        # To speed up, we can limit the number of iterations.
        
        def objective_direct(h_vec):
            if np.any(h_vec < 0):
                return 1e10
            return compute_score(h_vec)
            
        # Use a smaller maxiter for direct optimization to avoid timeout
        # L-BFGS-B with finite diff on 600 vars: each grad eval is 600 score evals.
        # 200 score evals is fast. 20 iterations -> 12000 score evals. 
        # Score eval is ~0.5ms. 12000 * 0.5ms = 6s. Feasible.
        
        try:
            res_direct = optimize.minimize(objective_direct, h_start, method='L-BFGS-B',
                                           bounds=[(0, None)] * N,
                                           options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-4})
            if res_direct.fun < best_score:
                best_score = res_direct.fun
                best_h = res_direct.x.copy()
        except Exception:
            pass
            
    return best_h.tolist()
