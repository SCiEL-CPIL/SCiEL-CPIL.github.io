import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def parametric_to_step(params, n_rects):
    """Convert parameters to a step function of length N.
    
    params: [center_0, width_0, height_0, center_1, width_1, height_1, ...]
    Each rectangle is non-overlapping (we'll handle overlaps by clipping).
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    f = np.zeros(N)
    for i in range(n_rects):
        c = params[3*i]
        w = abs(params[3*i+1]) + 0.001  # ensure positive width
        h = abs(params[3*i+2])  # ensure nonnegative height
        
        # Clip width to be within [-0.25, 0.25]
        w = min(w, 0.5)
        
        # Create rectangle
        mask = (x >= c - w/2) & (x <= c + w/2)
        f[mask] += h
    
    # Handle overlaps: if rectangles overlap, we sum them (which is fine for nonnegative)
    # But we want to encourage non-overlapping, so we could penalize overlaps
    # For now, just sum them
    
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_rects):
    """Objective function for differential evolution."""
    f = parametric_to_step(params, n_rects)
    return compute_score(f)

def local_refinement(h, start_time, max_time, alpha_init=0.05, threshold_frac=0.95):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (s * s)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity: for each bin j, how much does it contribute to peak lags?
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (s / s_new)  # keep same sum
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (s_new * s_new)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (s * s)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Very tight threshold for polish
        threshold = max_c * 0.99
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (s / s_new)
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (s_new * s_new)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.01, 0.1)
        else:
            alpha *= 0.95
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    t0 = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try differential evolution with different numbers of rectangles
    n_rects_options = [3, 4, 5, 6]
    
    for n_rects in n_rects_options:
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        
        # Define bounds for parameters
        # centers in [-0.2, 0.2], widths in [0.01, 0.3], heights in [0.1, 5.0]
        bounds = []
        for i in range(n_rects):
            bounds.append((-0.2, 0.2))  # center
            bounds.append((0.01, 0.3))  # width
            bounds.append((0.1, 5.0))   # height
        
        # Run differential evolution
        try:
            result = differential_evolution(
                lambda p: objective(p, n_rects),
                bounds,
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,
                workers=1
            )
            
            if result.success or result.fun < best_score:
                f_opt = parametric_to_step(result.x, n_rects)
                score_opt = compute_score(f_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_f = f_opt.copy()
                    
                    # Apply local refinement
                    refined_f, refined_score = local_refinement(
                        best_f, time.time(), TIME_LIMIT - 15
                    )
                    if refined_score < best_score:
                        best_score = refined_score
                        best_f = refined_f.copy()
                        
                        # Apply polish pass
                        polished_f, polished_score = polish_pass(
                            best_f, time.time(), min(10, TIME_LIMIT - 5)
                        )
                        if polished_score < best_score:
                            best_score = polished_score
                            best_f = polished_f.copy()
        except Exception as e:
            pass
    
    # Ensure nonnegativity and return
    best_f = np.maximum(best_f, 0.0)
    return best_f.tolist()
