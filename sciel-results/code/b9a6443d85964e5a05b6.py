import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 135  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    # Use FFT for faster convolution
    conv = np.fft.irfft(np.fft.rfft(f) * np.fft.rfft(f), n=2*N-1)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def objective_direct(f, beta=10.0):
    """
    Objective function for direct optimization on bin values.
    Uses soft-max surrogate for the peak autoconvolution.
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    conv = np.fft.irfft(np.fft.rfft(f) * np.fft.rfft(f), n=2*N-1)
    # Normalize by sum squared
    norm_conv = conv / (s * s)
    return smooth_max(norm_conv, beta)

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction.
    
    At each step, we identify lags where the autoconvolution is near its max,
    compute how much each bin contributes to those peaks, and reduce bins
    that contribute more to peaks than to off-peak regions.
    """
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
        
        conv = np.fft.irfft(np.fft.rfft(current_h) * np.fft.rfft(current_h), n=2*N-1)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity: for each bin j, how much does it contribute to peak lags?
        # sensitivity[j] = sum_{m in peaks} h[m-j] (with appropriate index wrapping/bounds)
        sensitivity = np.zeros(N)
        for m in peak_indices:
            # m is the lag index in [0, 2N-2]
            # Contribution to lag m comes from pairs (j, m-j) where 0<=j<N and 0<=m-j<N
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                # For each valid j, the term is h[j] * h[m-j]
                # We want the derivative wrt h[j], which is sum_{m} h[m-j] for peaks
                # We can accumulate this via convolution of indicator of peaks with h
                # But loop is fine for small number of peaks or we can vectorize
                indices = np.arange(j_min, j_max + 1)
                # h[m-j] for these j's
                # m-j ranges from m-j_min to m-j_max
                contrib = current_h[m - indices]
                sensitivity[indices] += contrib
        
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
        h_new = h_new * (N / s_new)
        
        conv_new = np.fft.irfft(np.fft.rfft(h_new) * np.fft.rfft(h_new), n=2*N-1)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.4)
        else:
            alpha *= 0.88
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
        
        conv = np.fft.irfft(np.fft.rfft(current_h) * np.fft.rfft(current_h), n=2*N-1)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
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
            if j_min <= j_max:
                indices = np.arange(j_min, j_max + 1)
                contrib = current_h[m - indices]
                sensitivity[indices] += contrib
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.fft.irfft(np.fft.rfft(h_new) * np.fft.rfft(h_new), n=2*N-1)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Direct Global Optimization
    # We use a subgradient/gradient approach on the direct bins.
    # Since N=600 is moderate, we can try a few restarts with different initializations.
    
    rng = np.random.RandomState(42)
    
    # Try a few different starting points for direct optimization
    # 1. Constant
    # 2. Random noise
    # 3. Smooth noise (low freq)
    
    initializations = [
        np.ones(N),
        rng.uniform(0.5, 1.5, N),
        np.exp(-np.linspace(-2, 2, N)**2), # Gaussian bump
        np.abs(np.sin(np.linspace(0, 10, N))), # Oscillatory
    ]
    
    for init_f in initializations:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
            
        # Normalize initial
        init_f = np.maximum(init_f, 0)
        s_init = np.sum(init_f)
        if s_init > 0:
            init_f = init_f * (N / s_init)
        
        # Use Nelder-Mead on a reduced set? No, N=600 is too high for NM.
        # Use a custom gradient descent loop with soft-max objective.
        # We will perform a simple projected gradient descent.
        
        f = init_f.copy()
        lr = 0.05
        beta = 5.0 # Start soft
        
        # Gradient of soft-max objective
        # Let G = f*f / S^2. Objective L = (1/beta) log sum exp(beta G_k)
        # dL/dG_k = exp(beta G_k) / sum exp(beta G_j)
        # dG_k/df_i = (2 * (f*f)_k * S^2 - G_k * 2 S * f_i) / S^4 
        #            = (2 * conv_k / S^2) - 2 * G_k * f_i / S
        #            = 2 * (conv_k / S^2 - G_k * f_i / S)
        
        # This is O(N^2) per gradient step if done naively, but we can use FFT for conv.
        # However, computing the full gradient vector is expensive.
        # Let's stick to the "sensitivity" method which is effectively a heuristic gradient.
        # It's much faster and worked well in the current program.
        
        # So, we skip heavy global gradient descent and rely on:
        # 1. Good initialization
        # 2. Aggressive local refinement (which acts as a local optimizer)
        
        # To improve on the current program's global search, we can try 
        # optimizing a small number of "control points" that define a spline, 
        # then expanding to N bins, then refining.
        
        # But the "island 3" finding said direct convex optimization worked.
        # Let's try a simple iterative reduction of peaks for a short duration.
        
        # We will run the local_refinement for a shorter time to act as a global-ish search
        # because it moves the shape significantly in early iterations.
        
        refined_f, refined_score = local_refinement(
            init_f, start_time, TIME_LIMIT - 30
        )
        
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score

    # Phase 2: Final Polish
    if time.time() - start_time < TIME_LIMIT - 10:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative and return list of floats
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
