import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score_fft(h):
    """
    Computes the normalized maximum of the autoconvolution using FFT.
    Metric: max(h*h) / (||h||_1 * ||h||_inf)
    """
    h = np.asarray(h, dtype=float)
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return float('inf')
        
    area = np.sum(h)
    max_val = np.max(h)
    
    if area == 0 or max_val == 0:
        return float('inf')
        
    # Use np.convolve for simplicity and correctness
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    return max_conv / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Parametric Search: Two-level step function centered
    # h_i = 1 if |i - center| < width/2 else alpha
    # We optimize width (integer) and alpha (float)
    
    best_h = None
    best_score = float('inf')
    
    # Grid search over width and alpha
    # Width from 1 to N
    # Alpha from 0 to 1
    
    center = N / 2.0
    idx = np.arange(N)
    
    # Coarse grid
    widths = np.linspace(1, N, 50).astype(int)
    alphas = np.linspace(0.0, 1.0, 50)
    
    for w in widths:
        if time.time() > deadline:
            break
        mask = np.abs(idx - center) < w / 2.0
        for a in alphas:
            h = np.where(mask, 1.0, a)
            if np.max(h) == 0: continue
            score = compute_score_fft(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
    if best_h is None:
        return [1.0] * N

    # Local Refinement on the best candidate
    # Use a simple hill climbing with adaptive steps
    current_h = best_h
    current_score = best_score
    
    np.random.seed(42)
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 100
    
    def perturb(h, scale):
        noise = np.random.normal(0, scale, N)
        h_new = h + noise
        h_new = np.clip(h_new, 0, 1)
        if np.max(h_new) == 0:
            return h, current_score
        return h_new, compute_score_fft(h_new)
    
    while time.time() < deadline:
        h_new, score_new = perturb(current_h, step_size)
        
        if score_new < current_score:
            current_h = h_new
            current_score = score_new
            no_improve_count = 0
            step_size = min(step_size * 1.1, 0.5)
        else:
            no_improve_count += 1
            step_size *= 0.95
            
            if no_improve_count > max_no_improve:
                # Try a larger jump
                h_jump, score_jump = perturb(current_h, 0.5)
                if score_jump < current_score:
                    current_h = h_jump
                    current_score = score_jump
                    no_improve_count = 0
                    step_size = 0.2
                else:
                    # Random restart from a new parametric point
                    w = np.random.randint(1, N)
                    a = np.random.rand()
                    mask = np.abs(idx - center) < w / 2.0
                    current_h = np.where(mask, 1.0, a)
                    current_score = compute_score_fft(current_h)
                    step_size = 0.2
                    no_improve_count = 0

    return list(current_h)
