import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _spectral_to_h(coeffs):
    """Convert spectral coefficients to height sequence.
    coeffs[0] is the DC offset (should be 0.5 for sum=N/2).
    coeffs[1] = a1 (cos fundamental amplitude)
    coeffs[2] = b1 (sin fundamental amplitude)
    coeffs[3] = a2, coeffs[4] = b2, etc.
    """
    n = len(coeffs)
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    # Pairs: (a_k, b_k) for k=1,2,...
    num_modes = (n - 1) // 2
    for k in range(1, num_modes + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        h += a * np.cos(2.0 * np.pi * k * i / N) + b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h):
    """Convert height sequence to spectral coefficients via DFT."""
    H = np.fft.rfft(h)
    coeffs = [H[0].real / N]  # DC component
    for k in range(1, len(H)):
        coeffs.append(2.0 * H[k].real / N)
        coeffs.append(-2.0 * H[k].imag / N)
    return np.array(coeffs)


def _fix_sum(h):
    """Project h onto the set {x in [0,1]^N : sum(x) = N/2} using bisection on a threshold."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # If already valid, return
    if np.all(h >= 0.0) and np.all(h <= 1.0) and abs(np.sum(h) - target) < 1e-14:
        return h
    
    # Bisection for the threshold t such that sum(clip(h - t, 0, 1)) = target
    # We want to find t such that sum(max(0, min(1, h_i - t))) = target
    # Note: this is equivalent to projecting onto the simplex with bounds [0,1]
    
    # The function f(t) = sum(max(0, min(1, h_i - t))) is monotonically decreasing in t
    # We need f(t) = target
    
    # Bounds for t:
    # If t <= min(h), f(t) = sum(h) - N*t
    # If t >= max(h), f(t) = 0
    # Actually, f(t) = sum(max(0, min(1, h_i - t)))
    
    # Let's think about this differently.
    # We want to find alpha such that sum(clip(h + alpha, 0, 1)) = target
    # g(alpha) = sum(clip(h + alpha, 0, 1)) is monotonically increasing in alpha
    
    # Find range of alpha
    # g(alpha) = 0 when h + alpha <= 0 for all i, i.e., alpha <= -max(h)
    # g(alpha) = N when h + alpha >= 1 for all i, i.e., alpha >= 1 - min(h)
    
    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0
    
    # Bisection
    for _ in range(60):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid
    
    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)
    
    # Fine-tune to get exact sum
    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0.0, 1.0)
    
    # Final correction
    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_final)
        else:
            idx = np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)
    
    return h_final


def _smooth_max_val_grad(h, beta):
    """Compute smooth max value and gradient w.r.t. h."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Gradient: grad = -(2/n) * sum_k w_k * (h_shifted)
    # cs[k] = (2/n) * (n/2 - r[k]) where r[k] = sum_i h[i]h[i+k]
    # d(cs[k])/d(h[j]) = -(2/n) * (h[j-k] + h[j+k]) (with wraparound)
    # grad[j] = sum_k w_k * d(cs[k])/d(h[j]) = -(2/n) * sum_k w_k * (h[j-k] + h[j+k])
    
    # Use FFT for efficiency
    # grad = -(2/n) * (w * h + w * h_flip) where * is circular convolution
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # Circular convolution w * h
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # For w * h_flip, note that h_flip[j] = h[-j mod n]
    h_flip = np.zeros(n)
    h_flip[0] = h[0]
    h_flip[1:] = h[1:][::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]
    
    grad = -(2.0 / n) * (conv_wh + conv_whf)
    return val, grad


def _spectral_optimize(num_modes, h0, beta, maxiter=500, ftol=1e-12):
    """Optimize in spectral space."""
    coeffs0 = _h_to_spectral(h0)
    # Truncate to num_modes + 1 coefficients (DC + num_modes pairs)
    n_coeff = 2 * num_modes + 1
    coeffs0 = coeffs0[:n_coeff]
    
    bounds = [(0.5 - 0.5, 0.5 + 0.5)]  # DC component free
    for _ in range(num_modes):
        bounds.append((-0.5, 0.5))
        bounds.append((-0.5, 0.5))
    
    def obj(c):
        h = _spectral_to_h(c)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        val, _ = _smooth_max_val_grad(h, beta)
        return val
    
    try:
        res = minimize(obj, coeffs0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': ftol, 'adaptive': True})
        h_opt = _spectral_to_h(res.x)
        h_opt = np.clip(h_opt, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)


def _projected_gradient_spectral(coeffs0, beta, max_iter=300, lr=0.01):
    """Projected gradient in spectral space with box+sum constraints on h."""
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    n_coeff = len(coeffs)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        h = _spectral_to_h(coeffs)
        h = _fix_sum(h)
        val, grad_h = _smooth_max_val_grad(h, beta)
        
        # Compute gradient w.r.t. spectral coefficients
        i = np.arange(N)
        grad_c = np.zeros(n_coeff)
        # DC component
        grad_c[0] = np.sum(grad_h) / N
        for k in range(1, (n_coeff - 1) // 2 + 1):
            a_idx = 2 * k - 1
            b_idx = 2 * k
            cos_k = np.cos(2.0 * np.pi * k * i / N)
            sin_k = np.sin(2.0 * np.pi * k * i / N)
            grad_c[a_idx] = (2.0 / N) * np.sum(grad_h * cos_k)
            grad_c[b_idx] = (2.0 / N) * np.sum(grad_h * sin_k)
        
        coeffs_new = coeffs - lr * grad_c
        # Project: ensure resulting h is in [0,1] and sum = N/2
        h_new = _spectral_to_h(coeffs_new)
        h_new = _fix_sum(h_new)
        coeffs = _h_to_spectral(h_new)[:n_coeff]
        
        if it % 50 == 49:
            lr = max(0.001, lr * 0.95)
    
    h_final = _spectral_to_h(coeffs)
    h_final = _fix_sum(h_final)
    return h_final, _score(h_final)


def _accelerated_smooth_max(h0, beta, max_iter=1000, lr_init=0.05):
    """Nesterov accelerated projected gradient descent in full space."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init
    
    best_val = _score(h)
    best_h = h.copy()
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        if it % 100 == 99:
            lr = max(0.001, min(0.1, lr * 0.95))
        
        if it % 50 == 49:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()
    
    return best_h, best_val


def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    i = np.arange(N)
    
    # Phase 1: Generate diverse structured starts
    starts = []
    
    # Random starts
    for _ in range(10):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))
    
    # Sine waves with various frequencies
    for f in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))
    
    # Cosine waves
    for f in [1, 2, 3]:
        for amp in [0.2, 0.3, 0.4]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))
    
    # Two-frequency combinations
    h_2f = 0.5 + 0.3 * np.cos(2 * np.pi * i / N) + 0.2 * np.cos(4 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))
    h_2f2 = 0.5 + 0.25 * np.sin(2 * np.pi * i / N) + 0.25 * np.sin(3 * np.pi * i / N)
    starts.append(_fix_sum(h_2f2))
    
    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)
    
    # Phase 2: Spectral optimization from best starts
    if best_h is not None:
        # Try different numbers of modes
        for num_modes in [2, 3, 4]:
            if time.time() > _DEADLINE - 20:
                break
            h_candidate, s_candidate = _spectral_optimize(num_modes, best_h, beta=50.0, maxiter=200)
            if s_candidate < best_score:
                _update_best(h_candidate)
    
    # Phase 3: Full-space Nesterov accelerated gradient with multiple betas
    if best_h is not None:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > _DEADLINE - 10:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta, max_iter=800, lr_init=0.05)
            if s_candidate < best_score:
                _update_best(h_candidate)
    
    # Phase 4: Final high-beta polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=400.0, max_iter=500, lr_init=0.02)
        if s_candidate < best_score:
            _update_best(h_candidate)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
