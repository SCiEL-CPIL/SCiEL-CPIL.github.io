def _subgrad_step(x, alpha):
    s = np.sum(x)
    if abs(s) < 1e-10: return x
    ac = _autoconv_fft(x)
    max_idx = np.argmax(np.abs(ac))
    max_val = ac[max_idx]
    
    # d(ac_k)/dx_j = x[j+k] + x[j-k]
    # We need to compute this vector for j=0..N-1
    # This is a correlation-like operation.
    # Let's just use the FFT method for the gradient of the specific peak?
    # No, for a single peak k, the gradient is a shifted version of x.
    
    # g_j = sign(max_val) * (x[j+k] + x[j-k]) (with 0 padding)
    g = np.zeros(N)
    sign_val = 1.0 if max_val > 0 else -1.0
    
    # x[j+k] is x shifted left by k (with wrap? No, linear conv)
    # If j+k < N, g[j] += x[j+k]
    # If j-k >= 0, g[j] += x[j-k]
    
    # Efficient way:
    # term1: x[j+k]. This is x[k:N] placed at 0:N-k, and 0 at N-k:N
    # term2: x[j-k]. This is x[0:N-k] placed at k:N, and 0 at 0:k
    # Wait, indices.
    # c_k = sum_{j=0}^{N-1-k} x_j x_{j+k}
    # d c_k / d x_j = x_{j+k} (if j+k < N) + x_{j-k} (if j-k >= 0)
    
    # Vectorized:
    # term1[j] = x[j+k] if j < N-k else 0
    # term2[j] = x[j-k] if j >= k else 0
    
    if k < N:
        g[:N-k] += x[k:]
        g[k:] += x[:N-k]
    
    g *= sign_val
    
    # Objective: f = max|ac| / s^2
    # grad f = (1/s^2) * grad max|ac| - (2 max|ac| / s^3) * 1
    
    grad_f = (1.0 / (s*s)) * g - (2.0 * abs(max_val) / (s**3)) * np.ones(N)
    
    return x - alpha * grad_f
