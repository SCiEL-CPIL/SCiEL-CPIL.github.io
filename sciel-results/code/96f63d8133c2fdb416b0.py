import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 100.0  # Safety margin
    
    # Initial guess: A Gaussian-like shape centered
    # We want to balance L2 concentration (peaky) vs L1 spread and convolution peak.
    # A Gaussian is a good candidate.
    
    def objective(params):
        h = np.maximum(params, 0) # Enforce non-negativity in the objective if bounds are not strictly respected
        # Actually, L-BFGS-B respects bounds. Let's use bounds.
        h1 = np.sum(h)
        h2_sq = np.sum(h**2)
        if h1 == 0 or h2_sq == 0:
            return 0.0
        conv = np.convolve(h, h, mode='full')
        conv_max = np.max(conv)
        if conv_max == 0:
            return 0.0
        # We want to MAXIMIZE h2_sq / (h1 * conv_max)
        # So we MINIMIZE h1 * conv_max / h2_sq
        return (h1 * conv_max) / h2_sq

    def gradient(params):
        # Numerical gradient is sufficient for N=50
        h = np.maximum(params, 0)
        h1 = np.sum(h)
        h2_sq = np.sum(h**2)
        if h1 == 0 or h2_sq == 0:
            return np.zeros(N)
        conv = np.convolve(h, h, mode='full')
        conv_max = np.max(conv)
        
        # We can use numerical gradient to be safe, or derive it.
        # Given N=50, numerical gradient with central differences is fast enough.
        eps = 1e-8
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            val_plus = (np.sum(h_plus) * np.max(np.convolve(h_plus, h_plus, mode='full'))) / np.sum(h_plus**2)
            
            h_minus = h.copy()
            h_minus[i] = max(0, h_minus[i] - eps)
            val_minus = (np.sum(h_minus) * np.max(np.convolve(h_minus, h_minus, mode='full'))) / np.sum(h_minus**2)
            
            if h[i] == 0:
                grad[i] = (val_plus - val_minus) / (2*eps)
            else:
                grad[i] = (val_plus - val_minus) / (2*eps)
        return grad

    # Multiple starts to avoid local minima
    best_score = 0.0
    best_h = None
    
    starts = [
        # Uniform
        np.ones(N),
        # Triangular
        np.abs(np.linspace(-1, 1, N)),
        # Gaussian centered
        np.exp(-0.5 * (np.linspace(-3, 3, N))**2),
        # Gaussian wider
        np.exp(-0.5 * (np.linspace(-2, 2, N))**2),
        # Gaussian narrower
        np.exp(-0.5 * (np.linspace(-4, 4, N))**2),
        # Sine bump
        np.sin(np.pi * np.linspace(0, 1, N))**2
    ]
    
    bounds = [(0, None)] * N
    
    for x0 in starts:
        if time.time() - start_time > max_time:
            break
            
        # Normalize start
        x0 = x0 / np.sum(x0)
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            
            # Calculate final score
            h1 = np.sum(h_opt)
            h2_sq = np.sum(h_opt**2)
            conv = np.convolve(h_opt, h_opt, mode='full')
            conv_max = np.max(conv)
            
            if h1 > 0 and h2_sq > 0 and conv_max > 0:
                score = h2_sq / (h1 * conv_max)
            else:
                score = 0.0
                
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception as e:
            pass

    # If optimization failed or was too slow, fallback to best start
    if best_h is None:
        # Fallback to Gaussian
        best_h = np.exp(-0.5 * (np.linspace(-3, 3, N))**2)
        best_h = best_h / np.sum(best_h)
        
    # Ensure return type is a list of floats
    return [float(x) for x in best_h]
