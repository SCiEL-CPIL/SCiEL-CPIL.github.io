import numpy as np
import time

N = 600
W = 1.0 / 1200.0

# EVOLVE-BLOCK-START
def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / l2_sq for step function h.
    Uses FFT on a fine grid for accurate peak detection.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    n_grid = 4096  # Increased resolution for better accuracy
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    # Map fine grid points to step function intervals
    # x_fine is in [-0.25, 0.25). Interval i is [-0.25 + i*W, -0.25 + (i+1)*W)
    # Index = floor((x + 0.25) / W)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid  # Next power of 2 >= 2*n_grid - 1
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    i = np.arange(N)
    t = i / N
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform baseline
    candidates.append(np.ones(N))
    
    # 2. Disjoint pulses (every other)
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    candidates.append(h_alt)
    
    # 3. Disjoint pulses (every 3rd)
    h_alt3 = np.zeros(N)
    h_alt3[0::3] = 1.0
    candidates.append(h_alt3)
    
    # 4. Cosine bell
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    
    # 5. Half cosine (zero at edges)
    candidates.append(np.maximum(0, np.cos(np.pi * (t - 0.5))))
    
    # 6. Gaussian-like
    for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
        candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))
    
    # 7. Triangular
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))
    
    # 8. Power law shapes
    for p in [0.5, 1.0, 2.0, 4.0]:
        base = 1 - np.abs(2 * t - 1)
        candidates.append(np.maximum(0, base ** p))
    
    # 9. Sinc-like
    candidates.append(np.sinc(2 * t - 1))
    
    # 10. Random smooth candidates
    np.random.seed(42)
    for _ in range(20):
        h_rand = np.random.rand(N)
        # Smooth it more aggressively
        kernel = np.ones(10) / 10
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        h_smooth = np.clip(h_smooth, 0, None)
        if np.sum(h_smooth) > 1e-15:
            candidates.append(h_smooth)
            
    # 11. Periodic patterns
    for freq in [10, 20, 50, 100]:
        candidates.append(0.5 + 0.5 * np.sin(2 * np.pi * freq * t))
        candidates.append(np.maximum(0, np.sin(2 * np.pi * freq * t)))
        
    # Evaluate all candidates
    best_h = np.ones(N)
    best_ratio = _compute_ratio_fast(best_h)
    
    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r = _compute_ratio_fast(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Local refinement using coordinate descent
    h = best_h.copy()
    step_size = max(0.01, np.max(h) * 0.01)
    
    for iteration in range(500):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        # Try perturbing a subset of coordinates
        n_trials = 100
        idxs = np.random.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h[idx]
            
            # Try increasing
            h[idx] = h_old + step_size
            r_plus = _compute_ratio_fast(h)
            
            # Try decreasing
            h[idx] = h_old - step_size
            r_minus = _compute_ratio_fast(h)
            
            h[idx] = h_old
            
            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio and h[idx] > 0:
                h[idx] = h_old - step_size
                best_ratio = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    # Final evaluation
    final_r = _compute_ratio_fast(h)
    if final_r < best_ratio:
        best_ratio = final_r
        best_h = h.copy()
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
