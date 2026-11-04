"""
Structural change: Direct subgradient descent on the true max-abs off-diagonal
autoconvolution metric, with coordinate-descent polish for lag-specific suppression.

Key insight: Instead of smoothing the max and optimizing phases in frequency domain,
we directly minimize max_k |c_k| / (sum x)^2 using:
1. Subgradient descent: identify the active lag(s) achieving the max, compute the
   gradient of |c_k| w.r.t. x, and take a step to reduce it.
2. Coordinate descent: for each active lag, adjust individual x_i to reduce |c_k|.
3. Sum constraint handled by projection (subtract mean to keep sum constant).

This avoids the smoothing approximation error and directly targets the worst lag.
"""
import numpy as np
import time

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _grad_abs_c_k(x, k):
    """
    Gradient of |c_k| w.r.t. x, where c_k is the k-th autoconvolution coefficient.
    c_k = sum_i x_i * x_{i+k} for k >= 0
    For k >= 0: d|c_k|/dx_i = sign(c_k) * (x_{i+k} if i+k < N else 0) + sign(c_k) * (x_{i-k} if i-k >= 0 else 0)
    We handle this via the convolution structure.
    """
    n = len(x)
    c = _autoconv(x)
    if k >= len(c):
        return np.zeros(n)
    sign_ck = np.sign(c[k])
    if sign_ck == 0:
        return np.zeros(n)
    
    # c_k = sum_{i=0}^{N-1-k} x_i * x_{i+k} for k >= 0
    # d(c_k)/dx_i = x_{i+k} (if i+k < N) + x_{i-k} (if i-k >= 0, for the symmetric term when k>0)
    # Actually for linear autoconvolution: c_k = sum_i x_i * x_{i+k} for i=0..N-1-k
    # So d(c_k)/dx_i = x_{i+k} if i+k < N, else 0
    # But also for negative lags... we only have k >= 1 here.
    
    grad = np.zeros(n)
    for i in range(n - k):
        grad[i] += x[i + k]
    # The symmetric contribution: when we write c_k = sum_i x_i x_{i+k}, 
    # varying x_j affects c_k through x_j * x_{j+k} (if j+k<N) and x_{j-k} * x_j (if j-k>=0)
    # Wait, c_k = sum_{i=0}^{N-1-k} x_i * x_{i+k}
    # d(c_k)/dx_j = x_{j+k} (if j+k < N, i.e., j < N-k) + x_{j-k} (if j-k >= 0, i.e., j >= k)
    # The second term comes from the fact that x_j also appears as x_{i+k} when i = j-k
    for i in range(k, n):
        grad[i] += x[i - k]
    
    return sign_ck * grad

def _subgradient_descent(x0, max_iter=2000, lr0=0.01, deadline=None):
    """
    Subgradient descent on the true max-abs off-diagonal metric.
    At each step, find the lag(s) achieving the max, compute subgradient, and step.
    """
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s  # normalize sum to 1
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = lr0
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        # Find all lags achieving (near) the max
        tol = max_c * 0.05
        active_lags = np.where(abs_c >= max_c - tol)[0]
        
        # Average subgradient over active lags
        grad = np.zeros(N)
        for k_idx in active_lags:
            k = k_idx + 1  # actual lag (1-indexed in c)
            g = _grad_abs_c_k(x, k)
            grad += g
        grad /= len(active_lags)
        
        # Metric = max_c / s^2, with s=1 after normalization
        # Gradient of metric w.r.t. x: grad / s^2 - 2*max_c * ones / s^3
        # But since we normalize s=1 each step, just use grad
        # Actually, we need to account for the constraint sum(x)=1
        # Project gradient: grad_proj = grad - mean(grad) * ones
        grad_proj = grad - np.mean(grad)
        
        x_new = x - lr * grad_proj
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.9995
    
    return best_x

def _coordinate_descent(x0, max_iter=500, deadline=None):
    """
    Coordinate descent: for each active lag, adjust individual x_i to reduce |c_k|.
    """
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        # Find the worst lag
        k_idx = np.argmax(abs_c)
        k = k_idx + 1
        
        # For this lag, try adjusting each x_j to reduce |c_k|
        # c_k = sum_{i=0}^{N-1-k} x_i * x_{i+k}
        # To reduce |c_k|, we want to decrease |sum x_i x_{i+k}|
        # Adjusting x_j by delta changes c_k by:
        #   delta * (x_{j+k} if j+k<N) + delta * (x_{j-k} if j-k>=0)
        #   = delta * (x_{j+k} + x_{j-k})  [with appropriate bounds]
        
        # Try small adjustments to random coordinates
        rng = np.random.default_rng(i * 7 + 1)
        for _ in range(50):
            j = rng.integers(0, N)
            coeff = 0.0
            if j + k < N:
                coeff += x[j + k]
            if j - k >= 0:
                coeff += x[j - k]
            
            if abs(coeff) < 1e-15:
                continue
            
            # To reduce |c_k|, adjust x_j in the direction that decreases |c_k|
            # d|c_k|/dx_j = sign(c_k) * coeff
            # So we want to move x_j in the direction -sign(c_k) * coeff
            direction = -np.sign(c[k]) * np.sign(coeff)
            
            # Try a few step sizes
            for step in [0.01, 0.005, 0.001]:
                x_test = x.copy()
                x_test[j] += direction * step
                s_test = np.sum(x_test)
                if abs(s_test) < 1e-12:
                    continue
                x_test = x_test / s_test
                val_test = _metric_offdiag(x_test)
                if val_test < best_val:
                    best_val = val_test
                    best_x = x_test.copy()
                    x = x_test.copy()
                    break
    
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Random sequences with various bias
    for bias in [0.3, 0.35, 0.4, 0.45, 0.5]:
        r = rng.random(N)
        x = np.where(r > bias, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
    
    # 3. Low-frequency dominant sequences
    for freq in [1, 2, 3, 5]:
        t = np.arange(N)
        x = np.cos(2 * np.pi * freq * t / N)
        x = x + 0.1 * rng.standard_normal(N)
        candidates.append(x)
    
    # 4. Golay-like: start with random ±1 and iteratively flip to reduce peak
    for trial in range(5):
        x = rng.choice([-1.0, 1.0], size=N)
        # Quick local search: flip random elements if it improves
        for _ in range(200):
            j = rng.integers(0, N)
            x_test = x.copy()
            x_test[j] = -x_test[j]
            if abs(np.sum(x_test)) > 5:
                val_test = _metric_offdiag(x_test)
                if val_test < _metric_offdiag(x):
                    x = x_test
        candidates.append(x)
    
    # Sort candidates by current metric
    scored = [(c, _metric_offdiag(c)) for c in candidates]
    scored.sort(key=lambda t: t[1])
    
    # Update best from initial candidates
    for x0, val in scored:
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Run subgradient descent on top candidates
    for x0, cur_val in scored[:8]:
        if time.time() > deadline - 40:
            break
        
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            x_opt = _subgradient_descent(x0, max_iter=1500, lr0=0.005, deadline=deadline - 30)
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Coordinate descent polish on the best
    if time.time() < deadline - 10:
        try:
            x_polished = _coordinate_descent(best_x, max_iter=300, deadline=deadline - 5)
            val = _metric_offdiag(x_polished)
            if val < best_val:
                best_val = val
                best_x = x_polished
        except Exception:
            pass
    
    # Final subgradient descent on the overall best
    if time.time() < deadline - 5:
        try:
            x_final = _subgradient_descent(best_x, max_iter=1000, lr0=0.002, deadline=deadline - 2)
            val = _metric_offdiag(x_final)
            if val < best_val:
                best_val = val
                best_x = x_final
        except Exception:
            pass
    
    return best_x
