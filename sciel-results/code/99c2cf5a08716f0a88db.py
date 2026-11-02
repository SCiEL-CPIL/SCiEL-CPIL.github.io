"""Erdos minimum overlap via spectral shaping + projected subgradient descent."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145  # safety margin before 150s kill


def _max_overlap(h):
    """Compute maximum overlap of h with shifted (1-h) using FFT."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    return np.max(corr)


def _soft_min_overlap(h, alpha=100.0):
    """Soft-min approximation of max overlap for smooth optimization."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    m = np.max(corr)
    # Log-sum-exp trick for numerical stability
    shifted = (corr - m) * alpha
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted)) / N) / alpha
    return log_sum_exp


def _soft_min_overlap_grad(h, alpha=100.0):
    """Gradient of soft-min overlap objective."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    m = np.max(corr)
    shifted = (corr - m) * alpha
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient with respect to h
    # corr[k] = sum_i h[i] * (1-h)[(i-k) mod N]
    # d corr[k] / d h[j] = (1-h)[(j-k) mod N] - h[j] * delta_{j-k,0}... 
    # Actually: corr[k] = sum_i h[i] * (1 - h[(i-k) mod N])
    # d corr[k] / d h[j] = (1 - h[(j-k) mod N]) - h[j] * (if j == i then derivative of (1-h[i-k]) w.r.t h[j] is -delta_{i-k,j})
    # Let me think more carefully.
    # corr[k] = sum_i h[i] * (1 - h[(i-k) % N])
    # d corr[k] / d h[j] = (1 - h[(j-k) % N]) + sum_i h[i] * d(1 - h[(i-k)%N])/d h[j]
    # = (1 - h[(j-k) % N]) - sum_i h[i] * delta_{(i-k)%N, j}
    # = (1 - h[(j-k) % N]) - h[(j+k) % N]
    # Wait, (i-k)%N = j => i = (j+k)%N
    # So d corr[k] / d h[j] = (1 - h[(j-k) % N]) - h[(j+k) % N]
    
    grad_corr = np.zeros((N, N))
    indices = np.arange(N)
    for k in range(N):
        # d corr[k] / d h[j] for all j
        # = (1 - h[(j-k) % N]) - h[(j+k) % N]
        term1 = 1.0 - h[(indices - k) % N]
        term2 = h[(indices + k) % N]
        grad_corr[k, :] = term1 - term2
    
    # Gradient of soft-min: sum_k weights[k] * d corr[k] / d h
    grad = weights @ grad_corr
    return grad


def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    diff = target - np.sum(h)
    
    # Iterative projection
    for _ in range(50):
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.minimum(h[mask] + step, 1.0)
        else:
            mask = h > 1e-12
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.maximum(h[mask] + step, 0.0)
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
    
    # Final exact adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        interior = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(interior):
            h[interior] += diff / np.sum(interior)
            h = np.clip(h, 0.0, 1.0)
    return h


def _spectral_init(freq_pattern=None):
    """
    Generate initial h from spectral shaping.
    We want low autocorrelation, so we design the spectrum of h to have
    flat magnitude at low frequencies (where autocorrelation peaks matter).
    """
    if freq_pattern is None:
        # Default: try to make |H[k]| as flat as possible for k near 0
        # Use a random phase with uniform magnitude
        mag = np.ones(N)
        mag[0] = N / 2.0  # DC component for mean = 0.5
        # Random phases
        rng = np.random.default_rng(12345)
        phases = rng.uniform(-np.pi, np.pi, N)
        # Ensure conjugate symmetry for real signal
        H = mag * np.exp(1j * phases)
        # Enforce conjugate symmetry
        for k in range(1, N // 2 + 1):
            H[N - k] = np.conj(H[k])
        h = np.fft.ifft(H).real
        h = h - np.mean(h) + 0.5
        h = np.clip(h, 0, 1)
        return _project(h)
    else:
        return _project(freq_pattern)


def _subgradient_descent(h0, max_iter=2000, alpha_init=0.01, time_limit=None, t_start=None):
    """Projected subgradient descent for soft-min overlap."""
    if t_start is None:
        t_start = time.time()
    
    h = np.asarray(h0, dtype=np.float64).copy()
    h = _project(h)
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    alpha = alpha_init
    soft_alpha = 100.0
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - t_start > time_limit:
            break
        
        # Compute objective and gradient
        obj = _soft_min_overlap(h, soft_alpha)
        grad = _soft_min_overlap_grad(h, soft_alpha)
        
        # Gradient step
        h_new = h - alpha * grad
        h_new = _project(h_new)
        
        # Check improvement
        val = _max_overlap(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive step size
        if it % 100 == 0 and it > 0:
            alpha *= 0.9
        # Increase soft_alpha over time for sharper max approximation
        soft_alpha = min(soft_alpha * 1.001, 500.0)
        
        # Convergence check
        if it > 100 and abs(_soft_min_overlap(h, soft_alpha) - obj) < 1e-12:
            break
    
    return best_h, best_val


def _bfgs_optimize(h0, max_iter=500, time_limit=None, t_start=None):
    """BFGS optimization using scipy."""
    if t_start is None:
        t_start = time.time()
    
    h0 = np.asarray(h0, dtype=np.float64).copy()
    h0 = _project(h0)
    
    def obj_and_grad(x):
        if time.time() - t_start > time_limit if time_limit else False:
            return _soft_min_overlap(x), np.zeros_like(x)
        f = _soft_min_overlap(x, alpha=150.0)
        g = _soft_min_overlap_grad(x, alpha=150.0)
        return f, g
    
    try:
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            h0,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': max_iter, 'maxfun': 5000}
        )
        h_opt = _project(result.x)
        val = _max_overlap(h_opt)
        return h_opt, val
    except Exception:
        return h0, _max_overlap(h0)


def construct():
    t0 = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform baseline
    inits.append(np.full(N, 0.5))
    
    # 2. Spectral shaping with random phases
    for seed in [42, 123, 456, 789, 101]:
        rng = np.random.default_rng(seed)
        mag = np.ones(N)
        mag[0] = N / 2.0
        phases = rng.uniform(-np.pi, np.pi, N)
        H = mag * np.exp(1j * phases)
        for k in range(1, N // 2 + 1):
            H[N - k] = np.conj(H[k])
        h = np.fft.ifft(H).real
        h = h - np.mean(h) + 0.5
        h = np.clip(h, 0, 1)
        inits.append(_project(h))
    
    # 3. Sinusoidal patterns
    for freq in [1, 2, 3, 4, 5, 7]:
        for A in [0.2, 0.3, 0.4, 0.5]:
            h = 0.5 + A * np.sin(2 * np.pi * freq * np.arange(N) / N)
            inits.append(_project(np.clip(h, 0, 1)))
    
    # 4. Cosine patterns
    for freq in [1, 2, 3, 5]:
        for A in [0.2, 0.3, 0.4]:
            h = 0.5 + A * np.cos(2 * np.pi * freq * np.arange(N) / N)
            inits.append(_project(np.clip(h, 0, 1)))
    
    # 5. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(15):
        h = rng.uniform(0, 1, N)
        inits.append(_project(h))
    
    # Evaluate all initializations
    best_h = None
    best_val = float('inf')
    
    for h0 in inits:
        val = _max_overlap(h0)
        if val < best_val:
            best_val = val
            best_h = h0.copy()
        
        if time.time() - t0 > DEADLINE - 30:
            break
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_val = _max_overlap(best_h)
    
    # Run subgradient descent from best
    remaining = DEADLINE - 10 - (time.time() - t0)
    if remaining > 5:
        h_opt, val_opt = _subgradient_descent(best_h, max_iter=1000, time_limit=remaining, t_start=t0)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Run BFGS from best
    remaining = DEADLINE - 5 - (time.time() - t0)
    if remaining > 3:
        h_opt, val_opt = _bfgs_optimize(best_h, max_iter=200, time_limit=remaining, t_start=t0)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Try a few more diverse starts with quick optimization
    diverse_starts = [
        np.array([(2*i + 1) / (2*N) for i in range(N)]),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.4 * np.cos(2 * np.pi * 2 * np.arange(N) / N),
    ]
    
    for h_start in diverse_starts:
        if time.time() - t0 > DEADLINE - 8:
            break
        
        h_start = _project(h_start)
        val_start = _max_overlap(h_start)
        
        if val_start < best_val + 0.5:
            remaining = DEADLINE - 5 - (time.time() - t0)
            if remaining > 2:
                h_opt, val_opt = _subgradient_descent(h_start, max_iter=500, time_limit=remaining, t_start=t0)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
    
    # Final polish with BFGS
    remaining = DEADLINE - 2 - (time.time() - t0)
    if remaining > 1:
        h_opt, val_opt = _bfgs_optimize(best_h, max_iter=100, time_limit=remaining, t_start=t0)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Ensure exact sum constraint
    final_h = _project(best_h)
    
    return final_h
