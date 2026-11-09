import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio(h):
    """Calculate the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Calculate autoconvolution
    # Use FFT for efficiency, though N=50 is small
    # Linear convolution length is 2N-1
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    C = np.fft.ifft(H * H).real
    # The convolution is symmetric. The peak is at the center.
    # For real input, the peak of autoconvolution is at index N-1? 
    # Actually, max(C) is sufficient.
    peak = np.max(C)
    if peak <= 0:
        return 0.0
    return l2_sq / (l1 * peak)

def _objective(y):
    """Objective function for minimization. h = y^2."""
    h = y**2
    # We want to minimize the inverse of the ratio
    # Ratio = l2_sq / (l1 * peak)
    # Minimize J = l1 * peak / l2_sq
    l1 = np.sum(h)
    if l1 == 0:
        return 1e10
    l2_sq = np.sum(h**2)
    if l2_sq == 0:
        return 1e10
    
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    C = np.fft.ifft(H * H).real
    peak = np.max(C)
    
    if peak <= 0:
        return 1e10
        
    return (l1 * peak) / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _ratio(best_h)
    
    # 1. Try parametric scans
    # Geometric decay
    for r in np.linspace(0.1, 0.95, 50):
        h = r ** np.arange(N)
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # Gaussian
    for sigma in np.linspace(1, 10, 50):
        x = np.arange(N)
        h = np.exp(-(x - N/2)**2 / (2*sigma**2))
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # 2. Local optimization from best point
    # Transform h to y = sqrt(h)
    y0 = np.sqrt(best_h)
    
    # We can also try random starts
    np.random.seed(42)
    starts = [y0]
    for _ in range(10):
        y_rand = np.random.rand(N)
        starts.append(y_rand)
        
    # Also add some structured random
    for _ in range(10):
        y_struct = np.zeros(N)
        # Sparse random
        indices = np.random.choice(N, size=10, replace=False)
        y_struct[indices] = 1.0
        starts.append(y_struct)

    for y_init in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective, y_init, method='Powell', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            if res.success or res.fun < 1e10:
                y_opt = res.x
                h_opt = y_opt**2
                s = _ratio(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Normalize? The ratio is scale invariant, so normalization doesn't change the score.
    # But returning very small or large numbers might cause issues?
    # The problem says "nonnegative heights".
    # Let's return the raw best_h.
    return list(best_h)
