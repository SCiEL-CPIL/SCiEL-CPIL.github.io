import time
import numpy as np

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fft_autoconv(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.fft(f, size)
    conv = np.fft.ifft(F * F).real[:2 * n - 1]
    return conv

def subgradient_descent(f, start_time, max_time, lr_init=0.5, decay=0.999):
    """Subgradient descent to minimize max autoconvolution.
    
    The gradient of max(conv) w.r.t. f_i is conv[f_i] (the i-th element of the
    cross-correlation, which equals conv[f_i] for autoconvolution).
    We normalize by sum(f)^2 to get the subgradient of the normalized objective.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    lr = lr_init
    n_iter = 0
    
    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_f = current_f * (N / s)
        
        conv = fft_autoconv(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Find peak indices
        peak_idx = np.argmax(conv)
        
        # Subgradient: for the peak at index m, the gradient w.r.t. f_j is f[m-j]
        # (for valid j). This is the contribution of each bin to the peak.
        j_min = max(0, peak_idx - (N - 1))
        j_max = min(N - 1, peak_idx)
        
        grad = np.zeros(N)
        grad[j_min:j_max+1] = current_f[peak_idx - j_max:peak_idx - j_min:-1]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad = grad / grad_norm
        
        # Update: move away from bins contributing to peak
        current_f = current_f - lr * grad
        current_f = np.maximum(current_f, 0.0)
        
        # Decay learning rate
        lr *= decay
        n_iter += 1
        
        # If learning rate is too small, stop
        if lr < 1e-10:
            break
    
    return best_f, best_val, n_iter

def coordinate_descent(f, start_time, max_time, n_sweeps=50):
    """Coordinate descent polish: optimize each bin independently.
    
    For each bin j, we can compute the optimal value that minimizes the max
    autoconvolution. This is a 1D optimization problem.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    no_improve_count = 0
    
    for sweep in range(n_sweeps):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for j in range(N):
            if time.time() - start_time > max_time:
                break
            
            # Save original value
            orig_val = current_f[j]
            
            # Try a range of values for bin j
            # The optimal value is likely near the current value
            # Try: 0, current/2, current, current*1.5, current*2, and some nearby values
            candidates = [0.0, orig_val * 0.25, orig_val * 0.5, orig_val * 0.75,
                         orig_val, orig_val * 1.25, orig_val * 1.5, orig_val * 2.0]
            
            # Also try some values based on the scale of other bins
            scale = np.median(current_f[current_f > 0]) if np.any(current_f > 0) else 1.0
            candidates.extend([scale * 0.1, scale * 0.5, scale, scale * 2.0])
            
            best_cand_val = orig_val
            best_cand_score = compute_score(current_f)
            
            for cand in candidates:
                current_f[j] = cand
                s = np.sum(current_f)
                if s < 1e-12:
                    current_f[j] = orig_val
                    continue
                # Normalize to sum = N
                test_f = current_f * (N / s)
                val = compute_score(test_f)
                if val < best_cand_score:
                    best_cand_score = val
                    best_cand_val = cand
            
            current_f[j] = best_cand_val
            if best_cand_val != orig_val:
                improved = True
            
            # Update best
            s = np.sum(current_f)
            if s < 1e-12:
                current_f = best_f.copy()
                break
            test_f = current_f * (N / s)
            val = compute_score(test_f)
            if val < best_val:
                best_val = val
                best_f = test_f.copy()
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
    
    return best_f, best_val

def sensitivity_refinement(f, start_time, max_time, alpha_init=0.05):
    """Sensitivity-based refinement from the inspiration."""
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_autoconv(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * 0.92
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity: for each bin j, how much does it contribute to peak lags?
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_f[m - j]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_f * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = h_new
            alpha = min(alpha * 1.03, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Subgradient descent with multiple restarts
    # Start from different initializations
    initializations = []
    
    # Constant
    initializations.append(np.ones(N))
    
    # Triangular bump
    x = np.linspace(-0.25, 0.25, N)
    tri = np.maximum(0, 1 - 4 * np.abs(x))
    initializations.append(tri)
    
    # Cosine bump
    cos_bump = np.maximum(0, np.cos(2 * np.pi * x))
    initializations.append(cos_bump)
    
    # Two bumps
    two_bumps = np.maximum(0, np.cos(2 * np.pi * (x - 0.1)) + np.cos(2 * np.pi * (x + 0.1)))
    initializations.append(two_bumps)
    
    # Random non-negative
    rng = np.random.RandomState(42)
    rand_init = np.abs(rng.randn(N))
    initializations.append(rand_init)
    
    for init_f in initializations:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        
        # Normalize
        s = np.sum(init_f)
        if s < 1e-12:
            continue
        init_f = init_f * (N / s)
        
        # Subgradient descent
        f_sg, val_sg, n_iter = subgradient_descent(
            init_f, start_time, TIME_LIMIT - 30, lr_init=0.5, decay=0.9995
        )
        
        if val_sg < best_score:
            best_score = val_sg
            best_f = f_sg.copy()
    
    # Phase 2: Sensitivity refinement
    if time.time() - start_time < TIME_LIMIT - 20:
        f_ref, val_ref = sensitivity_refinement(
            best_f, start_time, TIME_LIMIT - 10, alpha_init=0.05
        )
        if val_ref < best_score:
            best_score = val_ref
            best_f = f_ref.copy()
    
    # Phase 3: Coordinate descent polish
    if time.time() - start_time < TIME_LIMIT - 5:
        f_cd, val_cd = coordinate_descent(
            best_f, start_time, TIME_LIMIT - 2, n_sweeps=30
        )
        if val_cd < best_score:
            best_score = val_cd
            best_f = f_cd.copy()
    
    # Final: ensure non-negative and return as list
    best_f = np.maximum(best_f, 0.0)
    
    return list(best_f)

if __name__ == "__main__":
    f = construct()
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    val = np.max(conv) / (s * s)
    print(f"Score (discrete normalized): {val}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
