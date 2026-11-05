import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize with a sinusoidal pattern + small DC
    # h[i] = a + b * sin(2*pi*k*i/N)
    # Choose k=1 for lowest frequency
    a0 = 1.0
    b0 = 1.0
    k = 1
    i = np.arange(N)
    h0 = a0 + b0 * np.sin(2*np.pi*k*i/N)
    
    # Evaluate initial
    def objective(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2.0 * N * m / (s*s))
    
    def smooth_objective(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        # Soft max surrogate
        beta = 50.0
        m_smooth = np.logaddexp.reduce(conv / beta) * beta
        return abs(2.0 * N * m_smooth / (s*s))
    
    best_h = h0.copy()
    best_val = objective(h0)
    
    # Optimize
    try:
        result = minimize(smooth_objective, h0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() < deadline:
            if result.fun < best_val:
                # Polish with exact objective
                h_candidate = result.x
                v = objective(h_candidate)
                if v < best_val:
                    best_h = h_candidate
                    best_val = v
    except Exception:
        pass
    
    # Final check
    if time.time() < deadline - 5:
        # Try a few more initializations
        for k_try in [2, 3, 5]:
            if time.time() > deadline - 5:
                break
            h_init = a0 + b0 * np.sin(2*np.pi*k_try*i/N)
            try:
                res = minimize(smooth_objective, h_init, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if time.time() > deadline - 5:
                    break
                v = objective(res.x)
                if v < best_val:
                    best_h = res.x.copy()
                    best_val = v
            except Exception:
                pass
    
    return best_h
