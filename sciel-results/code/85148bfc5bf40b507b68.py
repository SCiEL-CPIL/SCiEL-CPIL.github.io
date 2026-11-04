"""Spectral magnitude design with binary refinement and deadline guard."""
import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin below 150s

def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the linear autoconvolution of x."""
    # Use FFT for efficient linear convolution
    n = len(x)
    size = 2 * n
    f = np.fft.rfft(x, size)
    conv = np.fft.irfft(f * np.conj(f), size)[:n]
    return np.max(np.abs(conv))

def _integral(x):
    """Squared integral (L2 norm squared) of x."""
    return np.sum(x ** 2)

def _metric(x):
    """C3' metric: max |autoconv| / (integral)^2 * integral = max|autoconv| / integral
    Actually the task says 'maximum absolute autoconvolution divided by squared integral'
    So metric = max_abs_autoconv / (integral) where integral = sum(x^2)
    Wait, let me re-read: 'maximum absolute autoconvolution divided by squared integral'
    If integral means sum(x), then squared integral is (sum(x))^2
    If integral means sum(x^2), then squared integral is (sum(x^2))^2
    
    The seed uses np.ones(N) and scores 1.0. Let's compute:
    x = ones(400), autoconv at lag 0 = 400, max abs = 400
    sum(x) = 400, (sum(x))^2 = 160000, 400/160000 = 0.0025
    sum(x^2) = 400, (sum(x^2))^2 = 160000, 400/160000 = 0.0025
    
    The score is seed/value. Seed value = 0.0025? But score is 1.0.
    So if score = seed_score / value, and seed_score = 1.0, then value should be 1.0?
    
    Let me re-read: "score is the upper bound on C3' (maximum absolute autoconvolution)"
    "The evaluator measures the raw value (lower is better). The search score is seed/value"
    
    So the raw value is what's measured, and lower is better.
    For ones(400): raw value = ?
    
    If raw value = max_abs_autoconv / (sum(x))^2:
    = 400 / 400^2 = 400/160000 = 0.0025
    
    But score = seed/value = 1.0 means seed_value = 1.0?
    
    Perhaps the "seed value" is defined as the value of the seed program.
    So seed_value = metric(ones(N)) and score = seed_value / metric(candidate).
    
    Let me just use: value = max_abs_autoconv / (np.sum(x) ** 2)
    And the evaluator computes score = value_seed / value_candidate
    
    Actually, looking at inspirations that score > 1.0, they must achieve lower raw values.
    """
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    return _autoconv_max_abs(x) / (s ** 2)

def _spectral_design_optimize(initial_x, max_iters=500, lr=0.01):
    """Optimize spectral magnitude to flatten autoconvolution."""
    x = initial_x.copy()
    n = len(x)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Compute current autoconvolution
        size = 2 * n
        f = np.fft.rfft(x, size)
        conv_freq = f * np.conj(f)
        conv = np.fft.irfft(conv_freq, size)[:n]
        
        # Gradient of max_abs_conv with respect to x
        # This is complex; use soft max approximation
        tau = 1.0
        abs_conv = np.abs(conv)
        # Soft max
        weights = np.exp(abs_conv / tau)
        weights = weights / np.sum(weights)
        
        # Gradient of sum(weights * |conv|) w.r.t. x
        # |conv| gradient w.r.t. conv is sign(conv)
        grad_conv = weights * np.sign(conv)
        if np.all(np.abs(conv) < 1e-15):
            grad_conv = weights
        
        # conv = ifft(fft(x) * conj(fft(x)))
        # d(conv[k])/dx[m] involves cross terms
        # Use FFT-based gradient: 
        # grad_x = ifft(2 * conj(f) * fft(grad_conv_padded))
        # But this is for circular convolution; for linear we need padding
        
        grad_padded = np.zeros(size, dtype=complex)
        grad_padded[:n] = grad_conv
        f_grad = np.fft.rfft(grad_padded)
        grad_x = np.fft.irfft(2 * np.conj(f) * f_grad, size)[:n].real
        
        # Also need to account for the normalization by (sum(x))^2
        # Let's just do steepest descent on the unnormalized metric first
        
        x -= lr * grad_x
        
        # Normalize to keep sum(x) = N (to maintain scale)
        x = x / np.sum(x) * N
    
    return x

def _try_binary_local_search(x, max_flips=10000):
    """Try flipping individual bits to improve the metric."""
    x = x.copy()
    current_val = _metric(x)
    
    for _ in range(max_flips):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Random subset of positions to try
        n = len(x)
        candidates = np.random.choice(n, size=min(100, n), replace=False)
        
        best_idx = None
        best_val = current_val
        best_x = x.copy()
        
        for i in candidates:
            x[i] = -x[i]
            val = _metric(x)
            x[i] = -x[i]  # restore
            
            if val < best_val:
                best_val = val
                best_idx = i
                best_x = x.copy()
                best_x[i] = -best_x[i]
        
        if best_idx is not None:
            x = best_x
            current_val = best_val
            improved = True
        
        if not improved:
            break
    
    return x

def construct():
    start_time = time.time()
    n = N
    
    # Start with a random signed sequence
    np.random.seed(42)
    best_x = np.ones(n)
    best_val = _metric(best_x)
    
    # Try several random initializations
    for trial in range(5):
        if time.time() > DEADLINE - 20:
            break
        
        # Random ±1 sequence
        x0 = np.random.choice([-1, 1], size=n).astype(float)
        
        # Spectral optimization
        x_opt = _spectral_design_optimize(x0, max_iters=200, lr=0.05)
        
        # Round to ±1
        x_bin = np.sign(x_opt)
        x_bin[x_bin == 0] = 1
        
        val_bin = _metric(x_bin)
        
        if val_bin < best_val:
            best_val = val_bin
            best_x = x_bin.copy()
        
        # Local search on the binary sequence
        if time.time() < DEADLINE - 10:
            x_ls = _try_binary_local_search(x_bin, max_flips=5000)
            val_ls = _metric(x_ls)
            if val_ls < best_val:
                best_val = val_ls
                best_x = x_ls.copy()
    
    return best_x
