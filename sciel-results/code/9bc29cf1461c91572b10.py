"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    dx = 0.5 / N  # interval width
    
    def compute_normalized_max(x):
        """Compute the normalized maximum of the autoconvolution."""
        x = np.asarray(x, dtype=float)
        # Ensure nonnegative
        x = np.maximum(x, 0)
        
        s = np.sum(x)
        if s < 1e-15:
            return 1e15
        
        # Linear autoconvolution via FFT
        n_fft = 2 * N
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real
        
        # The linear autoconvolution is in conv[0:N] (lags 0 to N-1)
        # For nonnegative x, the max should be at lag 0
        max_conv = np.max(conv[:N])
        
        norm = (s * dx)**2
        if norm < 1e-15:
            return 1e15
        
        return max_conv / norm
    
    # Try several initial shapes and pick the best one, then optimize
    
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Raised cosine (Hann window)
    n = np.arange(N)
    hann = 0.5 * (1 + np.cos(2 * np.pi * n / (N - 1)))
    candidates.append(hann)
    
    # 3. Gaussian-like
    center = (N - 1) / 2.0
    sigma = N / 4.0
    gauss = np.exp(-0.5 * ((n - center) / sigma)**2)
    candidates.append(gauss)
    
    # 4. Sinc-like (truncated)
    t = (n - center) / (N / 2.0)
    sinc = np.where(np.abs(t) < 1e-10, 1.0, np.sin(np.pi * t) / (np.pi * t))
    sinc = np.maximum(sinc, 0)
    candidates.append(sinc)
    
    # 5. Triangular
    tri = 1.0 - np.abs(n - center) / (center + 0.5)
    tri = np.maximum(tri, 0)
    candidates.append(tri)
    
    # 6. Flat-top with smooth edges
    flat_top = np.ones(N)
    edge = max(1, N // 10)
    for i in range(edge):
        frac = i / edge
        flat_top[i] = 0.5 * (1 + np.cos(np.pi * frac))
        flat_top[N - 1 - i] = 0.5 * (1 + np.cos(np.pi * frac))
    candidates.append(flat_top)
    
    # Evaluate all candidates
    best_x = None
    best_obj = np.inf
    for c in candidates:
        c = np.maximum(c, 0)
        obj = compute_normalized_max(c)
        if obj < best_obj:
            best_obj = obj
            best_x = c.copy()
    
    # Optimize the best candidate
    x0 = best_x.copy()
    
    def objective(x):
        return compute_normalized_max(x)
    
    bounds = [(0, None)] * N
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 10000, 'maxfun': 100000, 'gtol': 1e-10})
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        best_x = np.maximum(best_x, 0)
        return list(best_x)
    
    # Try additional optimization with different starting points
    for c in candidates:
        if time.time() > deadline - 5:
            break
        c = np.maximum(c, 0)
        try:
            result = minimize(objective, c, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 5000, 'maxfun': 50000, 'gtol': 1e-10})
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Local refinement with coordinate descent if time permits
    if time.time() < deadline - 5:
        step_size = 0.01 * max(1.0, np.max(best_x))
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Try small perturbations
                for delta in [step_size, -step_size]:
                    if best_x[i] + delta < 0:
                        continue
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    obj_trial = compute_normalized_max(x_trial)
                    if obj_trial < best_obj:
                        best_x = x_trial
                        best_obj = obj_trial
                        improved = True
    
    best_x = np.maximum(best_x, 0)
    return list(best_x)
# EVOLVE-BLOCK-END
