"""Autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

def _fft_convolve(h):
    """Full autoconvolution via FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    # Pad to next power of 2 for efficient FFT
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, n=fsize)
    conv = np.fft.irfft(H * H, n=fsize)[:size]
    return conv

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _fft_convolve(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _logsumexp(x, beta=50.0):
    """Soft-max surrogate: approximates max(x) smoothly."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    shifted = beta * (x - m)
    # clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    return m + np.log(np.sum(np.exp(shifted))) / beta

def _smooth_objective(h, beta=50.0):
    """Smooth objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _fft_convolve(h)
    soft_max = _logsumexp(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _smooth_grad(h, beta=50.0):
    """Gradient of smooth objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)
    conv = _fft_convolve(h)
    # Soft-max weights
    m = np.max(conv)
    shifted = np.clip(beta * (conv - m), -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # d(soft_max)/d(conv_k) = weights[k]
    # d(conv_k)/d(h_i) = h_{k-i} (for valid indices)
    # conv[k] = sum_j h[j]*h[k-j], so d(conv[k])/d(h[i]) = h[k-i]
    
    n = len(h)
    size = 2 * n - 1
    grad_conv = np.zeros(size)
    # For each k in conv, d(conv[k])/d(h[i]) = h[k-i] if 0<=k-i<n
    # Vectorized: for each i, the contribution to grad[i] is sum_k weights[k]*h[k-i]
    # This is a cross-correlation of weights with h
    # grad[i] = sum_k weights[k] * h[k-i]
    # = correlation of weights and h at lag i
    # Using FFT: this is the same as convolving weights with h reversed
    w_rev = weights[::-1]
    grad_conv_i = np.convolve(w_rev, h, mode='same')[:n]
    
    # Full gradient: d(2*N*soft_max/s^2)/d(h[i])
    # = 2*N * (d(soft_max)/d(h[i]) * s^2 - soft_max * 2*s) / s^4
    # = 2*N * (grad_conv_i * s - 2*soft_max) / s^3
    
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    grad = 2.0 * N * (grad_conv_i * s - 2.0 * soft_max) / (s * s * s)
    return grad

def _objective_exact(x):
    """Exact objective for final polish."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _fft_convolve(h)
    return 2.0 * N * np.max(conv) / (s * s)

def _make_smooth_initial(sigma_frac=0.3):
    """Gaussian bump, normalized to sum=1."""
    x = np.linspace(-1, 1, N)
    sigma = sigma_frac
    h = np.exp(-x**2 / (2 * sigma**2))
    h = h / np.sum(h)
    return h

def _make_fourier_initial(freq=2, phase=0.0, amp=0.3):
    """Low-frequency cosine modulation on a bump."""
    x = np.linspace(-1, 1, N)
    sigma = 0.35
    base = np.exp(-x**2 / (2 * sigma**2))
    mod = 1.0 + amp * np.cos(2 * np.pi * freq * x + phase)
    h = base * mod
    h = np.maximum(h, 0)  # keep positive for stability
    if np.sum(h) < 1e-12:
        h = base
    h = h / np.sum(h)
    return h

def _make_block_initial(n_blocks=4):
    """Block structure with smooth edges."""
    h = np.ones(N)
    block_size = N // n_blocks
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, N)
        # Smooth edge
        edge = min(10, block_size // 4)
        if start > 0 and i > 0:
            h[start:start+edge] = 0.5
        if end < N and i < n_blocks - 1:
            h[end-edge:end] = 0.5
    h = h / np.sum(h)
    return h

def _make_spectral_initial(freq=2, amp=0.5, offset=0.5):
    """Sine-wave with alternating-sign structure."""
    x = np.linspace(-1, 1, N)
    h = offset + amp * np.sin(2 * np.pi * freq * x)
    h = h / np.sum(h)
    return h

def _make_chebyshev_initial(order=3, amp=0.5):
    """Chebyshev polynomial with alternating signs."""
    x = np.linspace(-1, 1, N)
    # T_n(x) = cos(n * arccos(x))
    h = 1.0 + amp * np.cos(order * np.arccos(np.clip(x, -1, 1)))
    h = h / np.sum(h)
    return h

def _optimize_from(h0, beta_schedule=None, max_iter=200, tol=1e-8):
    """Run L-BFGS-B with soft-max objective and gradient, then polish with exact."""
    if beta_schedule is None:
        beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in beta_schedule:
        def obj(x):
            return _smooth_objective(x, beta)
        
        def grad(x):
            return _smooth_grad(x, beta)
        
        try:
            res = minimize(obj, h0, jac=grad, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
            h_new = res.x
            v = _value(h_new)
            if np.isfinite(v) and v < best_val:
                best_h = h_new.copy()
                best_val = v
            h0 = h_new  # warm start for next beta
        except Exception:
            pass
    
    # Final polish with exact objective
    try:
        res = minimize(_objective_exact, best_h, method='L-BFGS-B',
                      options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10})
        h_new = res.x
        v = _value(h_new)
        if np.isfinite(v) and v < best_val:
            best_h = h_new.copy()
            best_val = v
    except Exception:
        pass
    
    return best_h, best_val

def construct():
    t0 = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # Gaussian bumps with various widths
    for sigma in [0.15, 0.25, 0.3, 0.4, 0.5]:
        candidates.append(_make_smooth_initial(sigma))
    
    # Fourier-modulated bumps
    for freq in [1, 2, 3]:
        for phase in [0.0, np.pi/2, np.pi]:
            for amp in [0.2, 0.4, 0.6]:
                candidates.append(_make_fourier_initial(freq, phase, amp))
    
    # Block structures
    for nb in [2, 3, 4, 5, 6, 8]:
        candidates.append(_make_block_initial(nb))
    
    # Sine wave (positive part)
    x = np.linspace(-1, 1, N)
    for freq in [1, 2, 3]:
        h = np.sin(np.pi * freq * x)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-12:
            h = h / np.sum(h)
            candidates.append(h)
    
    # Spectral initializations with alternating signs
    for freq in [1, 2, 3, 4]:
        for amp in [0.3, 0.5, 0.7]:
            for offset in [0.5, 0.8, 1.0]:
                candidates.append(_make_spectral_initial(freq, amp, offset))
    
    # Chebyshev polynomials
    for order in [2, 3, 4, 5]:
        for amp in [0.3, 0.5, 0.7]:
            candidates.append(_make_chebyshev_initial(order, amp))
    
    # Random smooth (Gaussian process-like)
    rng = np.random.default_rng(42)
    for trial in range(5):
        h = rng.standard_normal(N)
        # Smooth with moving average
        kernel = np.ones(20) / 20
        h = np.convolve(h, kernel, mode='same')
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-12:
            h = h / np.sum(h)
            candidates.append(h)
    
    # Evaluate all candidates, keep top few
    scored = []
    for h in candidates:
        v = _value(h)
        if np.isfinite(v):
            scored.append((v, h))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top 10 diverse starts
    top_starts = [h for _, h in scored[:10]]
    
    best_h = np.ones(N) / N
    best_val = _value(best_h)
    
    # Optimize from each top start
    for i, h0 in enumerate(top_starts):
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 15:
            break
        
        # Use fewer iterations for later candidates to save time
        remaining = DEADLINE - 15 - elapsed
        if i < 4:
            max_iter = 150
        elif i < 7:
            max_iter = 80
        else:
            max_iter = 40
        
        h_opt, v_opt = _optimize_from(h0.copy(), max_iter=max_iter)
        
        if v_opt < best_val:
            best_val = v_opt
            best_h = h_opt
    
    # Final local polish from best with very high beta
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 5:
        try:
            # High-beta smooth polish
            res = minimize(lambda x: _smooth_objective(x, 800.0), best_h,
                          jac=lambda x: _smooth_grad(x, 800.0),
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            h_new = res.x
            v_new = _value(h_new)
            if np.isfinite(v_new) and v_new < best_val:
                best_h = h_new
                best_val = v_new
        except Exception:
            pass
        
        # Exact polish
        elapsed = time.time() - t0
        if elapsed < DEADLINE - 2:
            try:
                res = minimize(_objective_exact, best_h, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                h_new = res.x
                v_new = _value(h_new)
                if np.isfinite(v_new) and v_new < best_val:
                    best_h = h_new
                    best_val = v_new
            except Exception:
                pass
    
    return best_h

# Quick sanity check if run directly
if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Max conv: {np.max(np.convolve(h, h))}")
