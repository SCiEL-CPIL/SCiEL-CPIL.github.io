import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s

# Precompute FFT setup
K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Build fine-sampled function
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    # Normalized max: max_conv / (integral_f)^2
    score = max_conv / (integral_f ** 2)
    
    return score

def compute_score_fast(heights):
    """Faster version using precomputed structures for optimization."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return 1e10
    heights = np.maximum(heights, 0.0)
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * (0.5 / N)
    
    if integral_f <= 1e-10:
        return 1e10
    
    return max_conv / (integral_f ** 2)

def _objective(params):
    """Objective function for optimization: normalized max autoconvolution."""
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return 1e10
    
    heights = np.maximum(params, 0.0)
    return compute_score_fast(heights)

def _grad(params):
    """Gradient of the objective using finite differences (sparse)."""
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return np.zeros_like(params)
    
    f0 = _objective(params)
    grad = np.zeros_like(params)
    h = 1e-5
    
    # Use a subset of coordinates for gradient estimation to save time
    # Focus on coordinates that are likely to matter most
    indices = np.random.choice(N, size=min(100, N), replace=False)
    
    for idx in indices:
        params_p = params.copy()
        params_p[idx] += h
        fp = _objective(params_p)
        grad[idx] = (fp - f0) / h
    
    return grad

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform (known to give score 1.0)
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Strategy: Represent as sum of two disjoint rectangular blocks
    # This is a structural approach - the autoconvolution of a multi-rectangle
    # function is a sum of triangles whose peaks can be suppressed by
    # destructive interference of their support overlaps
    
    x = np.linspace(0, 1, N)
    
    # 1. First try the two-rectangle approach
    # Two rectangles: [a1, a2] and [b1, b2] with a2 < b1 (disjoint)
    # Heights: h1 on [a1,a2], h2 on [b1,b2], 0 elsewhere
    
    def two_rect_score(a1, a2, b1, b2, h1, h2):
        """Compute score for two disjoint rectangles."""
        heights = np.zeros(N)
        # Map positions to indices
        i1_start = int(a1 * N)
        i1_end = int(a2 * N)
        i2_start = int(b1 * N)
        i2_end = int(b2 * N)
        
        if i1_end <= i1_start or i2_end <= i2_start:
            return 1e10
        if i1_end >= i2_start:  # Overlapping
            return 1e10
        
        heights[i1_start:i1_end] = h1
        heights[i2_start:i2_end] = h2
        
        return compute_score_fast(heights)
    
    # Search over two-rectangle configurations
    # Parameters: positions and widths
    # a1, a2 = first rect (a1 < a2)
    # b1, b2 = second rect (b1 < b2, a2 < b1)
    # h1, h2 = heights
    
    # Grid search over key parameters
    best_two_rect_score = 1e10
    best_two_rect_h = None
    
    # Total width constraint: a2 - a1 + b2 - b1 should be reasonable
    # Try various configurations
    for w1_frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for w2_frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for gap_frac in [0.05, 0.1, 0.15, 0.2, 0.25]:
                for center_offset in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                    w1 = w1_frac
                    w2 = w2_frac
                    gap = gap_frac
                    total = w1 + gap + w2
                    if total > 0.9:
                        continue
                    
                    # Center the configuration
                    start = 0.5 - total / 2 + center_offset
                    a1 = start
                    a2 = start + w1
                    b1 = a2 + gap
                    b2 = b1 + w2
                    
                    if a1 < 0 or b2 > 1:
                        continue
                    
                    # Try different height ratios
                    for h1 in [0.5, 0.7, 1.0, 1.3, 1.5]:
                        for h2 in [0.5, 0.7, 1.0, 1.3, 1.5]:
                            if time.time() - START_TIME > TIME_LIMIT - 30:
                                break
                            
                            s = two_rect_score(a1, a2, b1, b2, h1, h2)
                            if s < best_two_rect_score:
                                best_two_rect_score = s
                                heights = np.zeros(N)
                                i1_start = int(a1 * N)
                                i1_end = int(a2 * N)
                                i2_start = int(b1 * N)
                                i2_end = int(b2 * N)
                                heights[i1_start:i1_end] = h1
                                heights[i2_start:i2_end] = h2
                                best_two_rect_h = heights.copy()
    
    if best_two_rect_h is not None and best_two_rect_score < best_score:
        best_score = best_two_rect_score
        best_h = best_two_rect_h.copy()
    
    # 2. Refine the best shape using scipy optimization
    if time.time() - START_TIME < TIME_LIMIT - 20:
        # Use the best shape as initial guess
        h_init = best_h.copy()
        
        # Try L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 200}
            )
            
            if result.success or result.fun < best_score:
                h_opt = np.maximum(result.x, 0.0)
                s_opt = compute_score_fast(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # 3. Try various smooth shapes as additional candidates
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
        h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
        h_cos = np.maximum(h_cos, 0.0)
        score = compute_score(h_cos)
        if score < best_score:
            best_score = score
            best_h = h_cos.copy()
    
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        h_plateau = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 0:
            ramp = np.linspace(0, 1, taper_len)
            h_plateau[:taper_len] = ramp
            h_plateau[-taper_len:] = ramp[::-1]
        score = compute_score(h_plateau)
        if score < best_score:
            best_score = score
            best_h = h_plateau.copy()
    
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3]:
        h_gauss = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        h_gauss = h_gauss / np.max(h_gauss)
        score = compute_score(h_gauss)
        if score < best_score:
            best_score = score
            best_h = h_gauss.copy()
    
    # 4. Random smooth perturbations around the best shape
    if time.time() - START_TIME < TIME_LIMIT - 10:
        rng = np.random.default_rng(42)
        for trial in range(20):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Create smooth perturbation using low-frequency Fourier modes
            g = np.zeros(N)
            num_modes = rng.integers(2, 6)
            for _ in range(num_modes):
                k = rng.integers(1, N // 4)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.05, 0.15)
                g += amp * np.cos(2 * np.pi * k * x + phase)
            
            g -= np.mean(g)
            
            for eps in [0.05, 0.1, 0.15, 0.2]:
                h_trial = best_h + eps * g
                h_trial = np.maximum(h_trial, 0.0)
                if np.sum(h_trial) <= 0:
                    continue
                score = compute_score(h_trial)
                if score < best_score:
                    best_score = score
                    best_h = h_trial.copy()
    
    # Final: try scaling
    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        h_scaled = best_h * scale
        score = compute_score(h_scaled)
        if score < best_score:
            best_score = score
            best_h = h_scaled.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
