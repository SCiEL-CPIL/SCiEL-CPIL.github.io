"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT (O(N log N) instead of O(N^2))."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * n - 1]

def _objective_hard(h):
    """Hard objective for final evaluation."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        return 2.0 * N * soft_max / (s * s)
    else:
        return 2.0 * N * m / (s * s)

def _soft_max_obj_grad(h, beta=50.0):
    """Objective and gradient for soft-max objective."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        soft_max = m + np.log(sum_exp) / beta
        
        # Gradient of soft_max w.r.t. conv
        grad_conv = exp_shifted / sum_exp
        
        # Gradient of conv w.r.t. h via FFT
        n_conv = len(conv)
        n_fft2 = 1
        while n_fft2 < n_conv + N - 1:
            n_fft2 *= 2
        GC = np.fft.rfft(grad_conv, n=n_fft2)
        HH = np.fft.rfft(h, n=n_fft2)
        # Cross-correlation: correlate(grad_conv, h)
        corr = np.fft.irfft(GC * np.conj(HH), n=n_fft2)[:n_conv]
        
        # grad_h[j] = 2 * sum_{k=0}^{N-1} grad_conv[j+k] * h[k]
        grad_h = np.zeros(N)
        for j in range(N):
            idx_start = j
            idx_end = min(j + N, n_conv)
            if idx_start < idx_end:
                grad_h[j] = 2.0 * np.dot(grad_conv[idx_start:idx_end], h[:idx_end - idx_start])
        
        val = 2.0 * N * soft_max / (s * s)
        # d(val)/d(h[j]) = 2*N/(s^2) * d(soft_max)/d(h[j]) - 2*N*soft_max/s^3 * 2*s * d(s)/d(h[j])
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * soft_max / (s * s * s)
        return val, grad_val
    else:
        val = 2.0 * N * m / (s * s)
        idx_max = np.argmax(conv)
        grad_h = np.zeros(N)
        for j in range(N):
            d = idx_max - j
            if 0 <= d < N:
                grad_h[j] = 2.0 * h[d]
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * m / (s * s * s)
        return val, grad_val

def _spectral_flatten_obj(h, lam=0.1):
    """Objective combining peak minimization with spectral flattening."""
    base_val = _soft_max_obj(h, beta=50.0)
    # Compute DCT coefficients and penalize their variance (flattening)
    baseline = np.mean(h)
    zero_mean = h - baseline
    # Use FFT as proxy for spectral content
    H = np.fft.rfft(zero_mean)
    power = np.abs(H) ** 2
    # Penalize peak-to-average ratio in spectrum
    total_power = np.sum(power)
    if total_power < 1e-10:
        return base_val
    max_power = np.max(power)
    avg_power = total_power / len(power)
    p2a = max_power / (avg_power + 1e-10)
    return base_val + lam * (p2a - 1.0)

def _spectral_flatten_grad(h, lam=0.1):
    """Objective and gradient for spectral flattening."""
    base_val, base_grad = _soft_max_obj_grad(h, beta=50.0)
    
    baseline = np.mean(h)
    zero_mean = h - baseline
    H = np.fft.rfft(zero_mean)
    power = np.abs(H) ** 2
    total_power = np.sum(power)
    if total_power < 1e-10:
        return base_val, base_grad
    
    max_power = np.max(power)
    avg_power = total_power / len(power)
    p2a = max_power / (avg_power + 1e-10)
    val = base_val + lam * (p2a - 1.0)
    
    # Gradient of spectral flattening term
    # d(p2a)/d(h) is complex, approximate via finite differences or skip
    # For now, just use base gradient and add a small regularization
    grad = base_grad.copy()
    # Add small L2 regularization to encourage smoothness
    grad += 0.001 * h
    return val, grad

def _random_walk_init(rng, N, drift=-0.01, noise_scale=0.5):
    """Generate random walk with negative drift as initialization."""
    h = np.zeros(N)
    for i in range(N):
        if i > 0:
            h[i] = h[i-1] + drift + noise_scale * rng.standard_normal()
        else:
            h[i] = noise_scale * rng.standard_normal()
    return h

def _coordinate_descent_polish(h, max_iter=500, deadline=None):
    """Polish using coordinate descent on the hard objective."""
    h = h.copy()
    best_val = _objective_hard(h)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for j in range(N):
            # Try small perturbations
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                h_test = h.copy()
                h_test[j] += delta
                val = _objective_hard(h_test)
                if val < best_val - 1e-10:
                    h = h_test
                    best_val = val
                    improved = True
                    break
        if not improved:
            break
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(5):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns
    for period in [2, 3, 4, 5]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2 * np.pi * freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2 * t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Polynomial bases
    for deg in [1, 2, 3]:
        x = np.linspace(-1, 1, N)
        h0 = np.zeros(N)
        for d in range(deg + 1):
            h0 += rng.standard_normal() * x**d
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b * block_size:(b + 1) * block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 6. DCT-shaped initializations
    from scipy.fftpack import dct, idct
    for num_modes in [5, 10, 20]:
        h0 = np.zeros(N)
        coeffs = np.zeros(N)
        for k in range(num_modes):
            coeffs[k] = rng.standard_normal() * (1.0 / (k + 1))
        zm = idct(coeffs, type=3, norm='ortho')
        h0 = 1.0 + zm
        initializations.append(h0)
    
    # 7. Random walk initializations
    for _ in range(5):
        h0 = _random_walk_init(rng, N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # Optimize each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
        
        h_curr = h0.copy()
        
        # Phase 1: Standard soft-max optimization
        if time.time() < deadline - 20.0:
            try:
                res = minimize(_soft_max_obj, h_curr,
                              method='L-BFGS-B',
                              args=(50.0,),
                              options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
                val = _objective_hard(res.x)
                if val < _objective_hard(h_curr):
                    h_curr = res.x
            except Exception:
                pass
        
        # Phase 2: Spectral flattening optimization
        if time.time() < deadline - 15.0:
            for lam in [0.05, 0.1, 0.2]:
                try:
                    res = minimize(_spectral_flatten_obj, h_curr,
                                  method='L-BFGS-B',
                                  args=(lam,),
                                  options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                    val = _objective_hard(res.x)
                    if val < _objective_hard(h_curr):
                        h_curr = res.x
                except Exception:
                    pass
        
        # Phase 3: Higher precision soft-max
        if time.time() < deadline - 10.0:
            for beta in [100.0, 200.0]:
                try:
                    res = minimize(_soft_max_obj, h_curr,
                                  method='L-BFGS-B',
                                  args=(beta,),
                                  options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                    val = _objective_hard(res.x)
                    if val < _objective_hard(h_curr):
                        h_curr = res.x
                except Exception:
                    pass
        
        # Update best
        val_curr = _objective_hard(h_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 15.0:
        for beta in [500.0, 1000.0]:
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Coordinate descent polish
    if time.time() < deadline - 5.0:
        try:
            h_cd = _coordinate_descent_polish(best_h, max_iter=200, deadline=deadline - 2.0)
            val_cd = _objective_hard(h_cd)
            if val_cd < best_val:
                best_val = val_cd
                best_h = h_cd
        except Exception:
            pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
