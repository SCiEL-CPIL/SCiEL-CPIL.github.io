import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value_exact(h):
    """
    Computes the exact objective value for the given heights h.
    Used for final evaluation and verification.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negative and finite
    if np.any(h < -1e-12) or np.any(~np.isfinite(h)):
        return 0.0
    
    # Clip tiny negatives from numerical noise
    h = np.maximum(h, 0.0)
    
    # FFT based convolution for speed and precision
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    
    # Theoretical g should be non-negative for autocorrelation of real signal?
    # Actually, g is autocorrelation, so g[k] = sum h[i]h[i+k]. Since h>=0, g>=0.
    # However, FFT numerical errors might introduce tiny negatives.
    # The task says h are nonnegative, so g must be nonnegative.
    g = np.maximum(g, 0.0)
    
    m = len(g)
    
    # l2 = (w/3) * sum_{i=0}^m (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    # y = [0, g[0], ..., g[m-1], 0]
    # Sum = g[0]^2 + (g[0]^2 + g[0]g[1] + g[1]^2) + ... + (g[m-2]^2 + g[m-2]g[m-1] + g[m-1]^2) + g[m-1]^2
    # = 2 * sum(g^2) + sum(g[i]g[i+1])
    
    g_sq_sum = np.dot(g, g)
    cross_sum = np.dot(g[:-1], g[1:]) if m > 1 else 0.0
    
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def compute_value_smooth(h, beta=100.0):
    """
    Computes a smooth approximation of the objective for optimization.
    Uses Log-Sum-Exp for the max term.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 1e-15) # Avoid log(0) or div by 0 issues in gradients if needed, though we use log_h
    
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    g = np.maximum(g, 0.0)
    
    m = len(g)
    
    g_sq_sum = np.dot(g, g)
    cross_sum = np.dot(g[:-1], g[1:]) if m > 1 else 0.0
    
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    
    sum_g = np.sum(g)
    norm_1 = sum_g / (m + 1)
    
    # Smooth max: (1/beta) * log(sum(exp(beta * g)))
    # To avoid overflow, shift by max
    g_max = np.max(g)
    if g_max < 1e-12:
        return 0.0
        
    # log_sum_exp(beta * g) = beta * g_max + log(sum(exp(beta * (g - g_max))))
    log_sum_exp = beta * g_max + np.log(np.sum(np.exp(beta * (g - g_max))))
    norm_inf_smooth = log_sum_exp / beta
    
    if norm_inf_smooth == 0 or norm_1 == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf_smooth)

def neg_value_smooth(log_h, beta=100.0):
    """
    Objective for L-BFGS-B.
    Input: log_h (vector of log heights)
    Output: -smooth_value
    """
    # Centering log_h helps with numerical stability and handles homogeneity
    # The objective is invariant to scaling h. 
    # If we scale h by c, g scales by c^2.
    # l2 scales by c^2. norm_1 scales by c^2. norm_inf scales by c^2.
    # value = (c^2 * l2_0) / (c^2 * norm_1_0 * c^2 * norm_inf_0) ?? No.
    # Let's check homogeneity.
    # g -> c^2 g.
    # l2 -> c^2 l2.
    # norm_1 -> c^2 norm_1.
    # norm_inf -> c^2 norm_inf.
    # value = (c^2 l2) / (c^2 norm_1 * c^2 norm_inf) = (1/c^2) * (l2 / (norm_1 norm_inf)).
    # Wait, the objective is NOT homogeneous of degree 0?
    # Let's re-read the formula.
    # value = l2 / (norm_1 * norm_inf)
    # l2 is quadratic in g. g is quadratic in h. So l2 is quartic in h.
    # norm_1 is linear in g -> quadratic in h.
    # norm_inf is linear in g -> quadratic in h.
    # Denominator is quartic in h.
    # So value is homogeneous of degree 0.
    # Yes, scaling h by c does not change the value.
    # Therefore, we can fix the scale. E.g., sum(log_h) = 0 or mean(log_h)=0.
    # Using centered log_h is a good strategy to remove the flat direction.
    
    centered_log_h = log_h - np.mean(log_h)
    h = np.exp(centered_log_h)
    
    val = compute_value_smooth(h, beta=beta)
    return -val

def neg_value_exact(log_h):
    """
    Objective for final polish using exact max.
    """
    centered_log_h = log_h - np.mean(log_h)
    h = np.exp(centered_log_h)
    val = compute_value_exact(h)
    return -val

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    best_h = np.ones(N)
    best_val = compute_value_exact(best_h)
    
    rng = np.random.RandomState(42)
    
    def try_optimize(start_log_h, maxiter, use_smooth=True, beta=100.0):
        nonlocal best_h, best_val
        t0 = time.time()
        
        if use_smooth:
            obj = lambda x: neg_value_smooth(x, beta=beta)
        else:
            obj = neg_value_exact
            
        try:
            res = minimize(
                obj,
                start_log_h,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            # Reconstruct h
            h_cand = np.exp(res.x - np.mean(res.x))
            h_cand = np.maximum(h_cand, 0.0)
            
            # Evaluate with EXACT formula to be sure
            v_cand = compute_value_exact(h_cand)
            
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
                
        except Exception:
            pass
            
        return time.time() - t0

    # 1. Start from uniform (log 0 = 0)
    try_optimize(np.zeros(N), maxiter=5000, use_smooth=True, beta=50.0)
    
    # 2. Multi-start perturbations
    # We generate diverse starting points in log-space
    n_starts = 25
    for i in range(n_starts):
        if time.time() > deadline - 40.0:
            break
            
        # Generate a random smooth-ish shape
        # Use a combination of random frequencies
        x = np.zeros(N)
        if i == 0:
            # Random smooth noise
            x = rng.randn(N)
            kernel = np.ones(5)/5.0
            x = np.convolve(x, kernel, mode='same')
        elif i == 1:
            # Sine wave
            x = np.sin(2 * np.pi * np.arange(N) / N)
        elif i == 2:
            # Cosine wave
            x = np.cos(2 * np.pi * np.arange(N) / N)
        elif i == 3:
            # Triangle wave
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
        elif i == 4:
            # Gaussian bump
            x = np.exp(-((np.arange(N)-N//2)**2)/10.0)
        elif i == 5:
            # Two bumps
            x = np.exp(-((np.arange(N)-N//4)**2)/5.0) + np.exp(-((np.arange(N)-3*N//4)**2)/5.0)
        elif i == 6:
            # Ramp
            x = np.linspace(-1, 1, N)
        elif i == 7:
            # Square wave
            x = np.sign(np.sin(2 * np.pi * np.arange(N) / N))
        elif i == 8:
            # High freq sine
            x = np.sin(4 * np.pi * np.arange(N) / N)
        elif i == 9:
            # Random spikes
            x = rng.randn(N)
        elif i == 10:
             # Smooth random with more high freq
             x = rng.randn(N)
             kernel = np.ones(3)/3.0
             x = np.convolve(x, kernel, mode='same')
        elif i == 11:
             # Asymmetric bump
             x = np.exp(-((np.arange(N)-N//3)**2)/8.0)
        elif i == 12:
             # Double sine
             x = np.sin(2 * np.pi * np.arange(N) / N) + 0.5 * np.sin(4 * np.pi * np.arange(N) / N)
        elif i == 13:
             # Plateau
             x = np.where((np.arange(N) > N//4) & (np.arange(N) < 3*N//4), 1, -0.5)
        elif i == 14:
             # Hann window shape
             x = 0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / N))
        elif i == 15:
             # Random smooth, heavier tail
             x = rng.randn(N)
             kernel = np.ones(7)/7.0
             x = np.convolve(x, kernel, mode='same')
        elif i == 16:
             # Sin^2
             x = np.sin(2 * np.pi * np.arange(N) / N)**2
        elif i == 17:
             # Cos^2
             x = np.cos(2 * np.pi * np.arange(N) / N)**2
        elif i == 18:
             # Random permutation of a smooth curve
             base = np.exp(-((np.arange(N)-N//2)**2)/15.0)
             perm = rng.permutation(N)
             x = base[perm]
        elif i == 19:
             # Smooth random, very low freq
             x = rng.randn(N)
             kernel = np.ones(11)/11.0
             x = np.convolve(x, kernel, mode='same')
        elif i == 20:
             # Sinc-like
             x = np.sinc(np.arange(N)/5.0)
        elif i == 21:
             # Comb
             x = (np.arange(N) % 5 == 0).astype(float)
             kernel = np.ones(3)/3.0
             x = np.convolve(x, kernel, mode='same')
        elif i == 22:
             # Random Fourier series
             freqs = rng.randint(1, 5, size=3)
             phases = rng.uniform(0, 2*np.pi, size=3)
             amps = rng.uniform(0.5, 1.5, size=3)
             t = np.arange(N)
             for f, p, a in zip(freqs, phases, amps):
                 x += a * np.sin(2 * np.pi * f * t / N + p)
        elif i == 23:
             # Exponential decay
             x = np.exp(-np.arange(N)/10.0)
        else:
             # Fallback random
             x = rng.randn(N)
             
        # Normalize/Center the starting point in log-space
        # We want log_h to be the starting point.
        # h_start = exp(x). So log_h_start = x.
        # But we need to ensure h_start is non-negative. exp(x) is always positive.
        # We also want to vary the scale slightly, but the optimizer handles it via centering.
        # However, different amplitudes might lead to different local optima if the landscape is not perfectly symmetric?
        # Since it's homogeneous, the scale doesn't matter for the value, but it might for the gradient direction in log-space?
        # Actually, in log-space, scaling h by c adds log(c) to all log_h.
        # The centering removes this degree of freedom.
        # So we just need diverse shapes.
        
        start_log_h = x.copy()
        
        # Optimize with smooth surrogate
        try_optimize(start_log_h, maxiter=3000, use_smooth=True, beta=100.0)
        
        # If time permits, do a quick polish with exact max on the result
        if time.time() < deadline - 20.0:
             # We don't know the current best log_h easily here, but we can re-optimize from the best_h
             # Actually, let's just run one more smooth step from the current best to refine
             pass

    # 3. Final Polish Phase
    # Use the best_h found so far to initialize a high-precision optimization
    if time.time() < deadline - 15.0:
        log_best = np.log(np.maximum(best_h, 1e-10))
        # Use a higher beta for smoother approximation closer to max
        try_optimize(log_best, maxiter=5000, use_smooth=True, beta=200.0)
        
    if time.time() < deadline - 5.0:
        # Exact max polish
        log_best = np.log(np.maximum(best_h, 1e-10))
        try_optimize(log_best, maxiter=2000, use_smooth=False)
        
    # One last check with a very high beta smooth to ensure we're at the "peak"
    if time.time() < deadline - 2.0:
        log_best = np.log(np.maximum(best_h, 1e-10))
        try_optimize(log_best, maxiter=1000, use_smooth=True, beta=500.0)

    return list(np.asarray(best_h, dtype=float))

# Dummy main to ensure module loads
if __name__ == "__main__":
    h = construct()
    print(f"Best Value: {compute_value_exact(h)}")
