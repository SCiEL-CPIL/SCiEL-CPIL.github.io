import time
import numpy as np
from scipy import optimize

N = 600

def compute_score(h):
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with uniform as baseline
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    # Two-block parameterization: h[i] = 1 if interval center in block1 or block2
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * (0.5 / N)  # interval centers
    
    def param_two_blocks(params):
        c1, w1, c2, w2 = params
        h = np.zeros(N)
        mask1 = (x >= c1 - w1) & (x <= c1 + w1)
        mask2 = (x >= c2 - w2) & (x <= c2 + w2)
        h[mask1] = 1.0
        h[mask2] = 1.0
        return h
    
    def objective(params):
        h = param_two_blocks(params)
        if np.sum(h) < 1e-12:
            return 1.0
        return compute_score(h)
    
    # Try multiple initial guesses for two symmetric blocks
    initial_guesses = [
        [0.20, 0.08, -0.20, 0.08],
        [0.18, 0.10, -0.18, 0.10],
        [0.15, 0.12, -0.15, 0.12],
        [0.22, 0.06, -0.22, 0.06],
        [0.10, 0.15, -0.10, 0.15],
        [0.24, 0.04, -0.24, 0.04],
    ]
    
    bounds = [(-0.25, 0.25), (0.001, 0.25), (-0.25, 0.25), (0.001, 0.25)]
    
    for guess in initial_guesses:
        if time.time() > deadline - 30:
            break
        try:
            res = optimize.minimize(objective, guess, method='Nelder-Mead',
                                     bounds=bounds,
                                     options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 800})
            if res.fun < best_score:
                best_score = res.fun
                best_h = param_two_blocks(res.x)
        except Exception:
            pass
    
    # Coordinate descent refinement on individual heights
    # Focus on reducing the peak of the autoconvolution
    for outer in range(8):
        if time.time() > deadline - 10:
            break
        
        conv = np.convolve(best_h, best_h, mode='full')
        k_peak = np.argmax(conv)
        
        # For each index, try reducing h[i] if it contributes to the peak
        for i in range(N):
            if time.time() > deadline - 10:
                break
            if best_h[i] < 1e-8:
                continue
            
            # Try reducing h[i] by 50%
            h_new = best_h.copy()
            h_new[i] *= 0.5
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            conv_new = np.convolve(h_new, h_new, mode='full')
            score_new = np.max(conv_new) / (s_new * s_new)
            
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_h = h_new
        
        # Also try small increases on zeros to fill gaps that might reduce peak
        for i in range(N):
            if time.time() > deadline - 10:
                break
            if best_h[i] > 1e-8:
                continue
            
            h_new = best_h.copy()
            h_new[i] = 0.1
            s_new = np.sum(h_new)
            conv_new = np.convolve(h_new, h_new, mode='full')
            score_new = np.max(conv_new) / (s_new * s_new)
            
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_h = h_new
    
    return best_h.tolist()
