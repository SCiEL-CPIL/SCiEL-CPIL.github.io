import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconv(h):
    """Fast autoconvolution via FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    l_inf = np.max(np.abs(conv))
    if l_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _neg_log_ratio(h):
    """Negative log of the ratio for smooth optimization."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 1e-12)
    l1 = np.sum(h)
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    l_inf = np.max(np.abs(conv))
    if l_inf < 1e-15 or l1 < 1e-15:
        return 1e10
    return -np.log(l2_sq / (l1 * l_inf))

def _neg_log_ratio_jac(h):
    """Gradient of negative log ratio."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 1e-12)
    l1 = np.sum(h)
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    l_inf = np.max(np.abs(conv))
    if l_inf < 1e-15 or l1 < 1e-15:
        return np.zeros(N)
    
    # Gradient of log(l2_sq) = 2*h / l2_sq
    grad_l2 = 2.0 * h / l2_sq
    # Gradient of log(l1) = 1 / l1
    grad_l1 = 1.0 / l1
    # Gradient of log(l_inf): need to find which index achieves max
    # This is non-smooth, so use a soft approximation
    # For simplicity, use subgradient: pick the argmax index
    idx_max = np.argmax(np.abs(conv))
    # The convolution at idx_max is sum_j h_j * h[idx_max - j] (with appropriate bounds)
    # Gradient of conv[idx_max] w.r.t. h_k is h[idx_max - k] + h[k - idx_max] (symmetric)
    # Actually for autoconvolution (h*h)[m] = sum_j h_j h_{m-j}
    # d/dh_k (h*h)[m] = h_{m-k} + h_{k-m} (if we consider both terms)
    # But since h*h is symmetric and we're doing full convolution, 
    # (h*h)[m] = sum_j h_j * h[m-j]
    # d/dh_k = h[m-k] + h[k-m] when both indices are valid
    
    grad_l_inf = np.zeros(N)
    m = idx_max
    for k in range(N):
        # h[m-k] if 0 <= m-k < N
        if 0 <= m - k < N:
            grad_l_inf[k] += h[m - k]
        # h[k-m] if 0 <= k-m < N
        if 0 <= k - m < N:
            grad_l_inf[k] += h[k - m]
    
    # d/dh_k [-log(l2_sq) + log(l1) + log(l_inf)]
    # = -grad_l2 + grad_l1 + grad_l_inf / l_inf
    jac = -grad_l2 + grad_l1 + grad_l_inf / l_inf
    return jac

def _polish(h, deadline):
    """Polish a candidate using scipy minimize with trust-constr."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    # Normalize to avoid numerical issues
    if np.sum(h) < 1e-15:
        return h, _ratio(h)
    
    try:
        res = minimize(
            _neg_log_ratio,
            h,
            jac=_neg_log_ratio_jac,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12}
        )
        if time.time() > deadline - 5:
            return h, _ratio(h)
        
        h_opt = np.maximum(res.x, 0.0)
        s_opt = _ratio(h_opt)
        
        if s_opt > _ratio(h):
            return h_opt, s_opt
        return h, _ratio(h)
    except:
        return h, _ratio(h)

def _generate_starts(deadline):
    """Generate diverse starting points."""
    starts = []
    
    # Uniform
    starts.append(np.ones(N))
    
    # Gaussian bumps at various centers and widths
    for center in [N//4, N//2, 3*N//4]:
        for width in [3, 5, 8, 12, 15]:
            x = np.arange(N)
            h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            starts.append(h)
    
    # Triangular
    for center in [N//4, N//2, 3*N//4]:
        for width in [5, 10, 15]:
            x = np.arange(N)
            h = np.maximum(0, 1 - np.abs(x - center) / width)
            starts.append(h)
    
    # Two peaks
    for c1 in [N//4, N//3]:
        for c2 in [3*N//4, 2*N//3]:
            for w in [3, 5, 8]:
                x = np.arange(N)
                h = np.exp(-((x - c1) ** 2) / (2 * w ** 2)) + np.exp(-((x - c2) ** 2) / (2 * w ** 2))
                starts.append(h)
    
    # Random sparse
    np.random.seed(42)
    for _ in range(20):
        h = np.zeros(N)
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h[idx] = np.random.uniform(0.1, 1.0, k)
        starts.append(h)
    
    # Random smooth
    for _ in range(20):
        x = np.arange(N)
        # Random Fourier series
        h = np.ones(N)
        for freq in [1, 2, 3, 4, 5]:
            amp = np.random.uniform(0, 0.3)
            phase = np.random.uniform(0, 2*np.pi)
            h += amp * np.cos(2*np.pi*freq*x/N + phase)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-10:
            starts.append(h)
    
    # Plateau shapes
    for left in range(0, N-5, 3):
        for right in range(left+5, N, 3):
            h = np.zeros(N)
            h[left:right] = 1.0
            starts.append(h)
    
    # Tapered plateau
    for left in range(0, N-8, 3):
        for right in range(left+8, N, 3):
            h = np.zeros(N)
            n_plateau = right - left
            taper = min(3, n_plateau // 3)
            h[left:left+taper] = np.linspace(0, 1, taper)
            h[left+taper:right-taper] = 1.0
            h[right-taper:right] = np.linspace(1, 0, taper)
            starts.append(h)
    
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Generate and evaluate starts
    starts = _generate_starts(deadline)
    
    for h0 in starts:
        if time.time() > deadline - 30:
            break
        
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 0.0)
        s0 = _ratio(h0)
        
        if s0 > best_score:
            best_score = s0
            best_h = h0.copy()
        
        # Polish if promising
        if s0 > best_score * 0.8 or time.time() < deadline - 60:
            h_polished, s_polished = _polish(h0, deadline)
            if s_polished > best_score:
                best_score = s_polished
                best_h = h_polished.copy()
    
    # Final polish on the best
    if time.time() < deadline - 10:
        h_final, s_final = _polish(best_h, deadline)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    return list(best_h)

if __name__ == "__main__":
    uniform = np.ones(N)
    print("Uniform score:", _ratio(uniform))
    
    result = construct()
    print("Best score:", _ratio(np.array(result)))
    print("Result sum:", sum(result))
