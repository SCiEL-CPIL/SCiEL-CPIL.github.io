import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)  # enforce nonnegativity
    norm1 = np.sum(h)
    if norm1 < 1e-12:
        return 0.0
    norm2_sq = np.sum(h**2)
    # Autoconvolution: h*h
    # For discrete sequences, (h*h)[k] = sum_i h[i] * h[i+k]
    # Using FFT for efficiency
    n_conv = 2 * N
    h_padded = np.zeros(n_conv)
    h_padded[:N] = h
    conv = np.fft.irfft(np.fft.rfft(h_padded) ** 2)
    # conv[k] for k=0..N-1 gives the autoconvolution
    sup_conv = np.max(np.abs(conv[:N]))
    if sup_conv < 1e-12:
        return 0.0
    return norm2_sq / (norm1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_h = np.ones(N)
    best_ratio = _ratio(best_h)
    
    # Normalize to ||h||_1 = 1 for optimization
    def _objective(x):
        h = np.exp(x)  # ensure positivity
        h = h / np.sum(h)  # normalize to ||h||_1 = 1
        return -_ratio(h)
    
    # Try multiple random restarts
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initial point
        x0 = np.random.randn(N) * 0.5
        h0 = np.exp(x0)
        h0 = h0 / np.sum(h0)
        
        # Gradient-free optimization using Nelder-Mead
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            h_opt = np.exp(result.x)
            h_opt = h_opt / np.sum(h_opt)
            r = _ratio(h_opt)
            if r > best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Also try structured patterns
    patterns = []
    
    # Uniform
    patterns.append(np.ones(N) / N)
    
    # Two-point concentration
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            patterns.append(h)
    
    # Three-point
    for i in range(0, N, 5):
        for j in range(i+1, N, 5):
            for k in range(j+1, N, 5):
                h = np.zeros(N)
                h[i] = 1/3
                h[j] = 1/3
                h[k] = 1/3
                patterns.append(h)
    
    # Sparse random
    for _ in range(100):
        h = np.zeros(N)
        idx = np.random.choice(N, size=np.random.randint(2, 10), replace=False)
        h[idx] = 1.0
        h = h / np.sum(h)
        patterns.append(h)
    
    for h in patterns:
        if time.time() > deadline:
            break
        r = _ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Local refinement around best
    if time.time() < deadline - 10:
        def _objective_local(h_vec):
            h = np.maximum(h_vec, 0)
            s = np.sum(h)
            if s < 1e-12:
                return 0.0
            h = h / s
            return -_ratio(h)
        
        # Use SLSQP with bounds
        bounds = [(0, 1)] * N
        try:
            result = minimize(_objective_local, best_h, method='SLSQP',
                             bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            h_opt = np.maximum(result.x, 0)
            s = np.sum(h_opt)
            if s > 1e-12:
                h_opt = h_opt / s
                r = _ratio(h_opt)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Return as list of nonnegative heights
    return [float(x) for x in best_h]
