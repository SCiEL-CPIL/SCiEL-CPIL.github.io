import numpy as np
import time

N = 600

def compute_score(h):
    """Compute the normalized maximum of autoconvolution."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def softplus(x):
        return np.where(x > 30, x, np.log1p(np.exp(x)))
    
    def softplus_grad(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    
    np.random.seed(42)
    x = np.zeros(N)
    best_h = softplus(x).copy()
    best_score = compute_score(best_h)
    
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    
    fft_size = 2 * N
    
    while time.time() < deadline:
        t += 1
        h = softplus(x)
        area = np.sum(h)
        max_val = np.max(h)
        
        if area == 0 or max_val == 0:
            break
        
        # Compute autoconvolution via FFT
        f = np.zeros(fft_size)
        f[:N] = h
        F = np.fft.fft(f)
        conv = np.fft.ifft(F * F).real
        max_conv = np.max(conv)
        k_max = np.argmax(conv)
        
        score = max_conv / (area * max_val)
        
        if not np.isfinite(score):
            break
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Gradient of max_conv w.r.t. h
        # d(conv[k])/d(h[i]) = 2*h[k-i] if 0<=k-i<N else 0
        j_idx = k_max - np.arange(N)
        valid = (j_idx >= 0) & (j_idx < N)
        grad_conv = np.zeros(N)
        grad_conv[valid] = 2.0 * h[j_idx[valid]]
        
        # d(area)/d(h[i]) = 1
        # d(max_val)/d(h[i]) subgradient
        is_max = (h == max_val)
        n_max = np.sum(is_max)
        grad_max_val = is_max / n_max if n_max > 0 else np.zeros(N)
        
        denom = (area * max_val) ** 2
        grad_h = (grad_conv * area * max_val - max_conv * (max_val + area * grad_max_val)) / denom
        grad_x = grad_h * softplus_grad(x)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_x
        v = beta2 * v + (1 - beta2) * grad_x ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Periodic restarts after 60s
        if t % 200 == 0 and time.time() > start_time + 60:
            x_new = np.random.normal(0, 1, N)
            h_new = softplus(x_new)
            score_new = compute_score(h_new)
            if score_new < best_score:
                x = x_new
                m = np.zeros(N)
                v = np.zeros(N)
                t = 0
    
    h_final = softplus(x)
    score_final = compute_score(h_final)
    if score_final < best_score:
        best_h = h_final
    
    return list(best_h)
