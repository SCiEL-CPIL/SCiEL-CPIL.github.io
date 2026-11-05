"""Combined program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

def _autoconv_fft(h):
    """Fast autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(f * f, n=size)
    return conv

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(x, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    h = x
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

def _soft_max_grad(x, beta=50.0):
    """Gradient of soft-max objective."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m <= 0:
        # Gradient of 2*N*m/(s*s) where m is max of conv
        # This is tricky - use subgradient approach
        idx = np.argmax(conv)
        # d(conv[idx])/d(h[j]) = h[idx-j] if idx-j >= 0 else h[idx+N-1-j]
        grad_conv = np.zeros_like(h)
        for j in range(N):
            k = idx - j
            if 0 <= k < N:
                grad_conv[j] = h[k]
            else:
                k2 = idx + N - 1 - j
                if 0 <= k2 < N:
                    grad_conv[j] = h[k2]
        ds = np.ones_like(h)
        return 2.0 * N * (grad_conv / (s * s) - 2.0 * m * ds / (s * s * s))
    else:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        # soft_max = m + log(sum_exp)/beta
        # d(soft_max)/d(conv[i]) = exp_shifted[i] / sum_exp / beta
        weights = exp_shifted / sum_exp / beta
        # d(soft_max)/d(h[j]) = sum_i weights[i] * d(conv[i])/d(h[j])
        # conv[i] = sum_k h[k] * h[i-k] (with appropriate bounds)
        # d(conv[i])/d(h[j]) = h[i-j] if i-j in [0,N-1] else 0 (for full conv)
        # Actually for full conv: conv[i] = sum_{k=0}^{N-1} h[k]*h[i-k] where h is extended with zeros
        # d(conv[i])/d(h[j]) = h[i-j] + h[j-i] (symmetric)
        grad = np.zeros_like(h)
        for i in range(2 * N - 1):
            w = weights[i]
            if w < 1e-15:
                continue
            for j in range(N):
                k1 = i - j
                if 0 <= k1 < N:
                    grad[j] += w * h[k1]
                k2 = j - i
                if 0 <= k2 < N:
                    grad[j] += w * h[k2]
        # Now multiply by 2*N/(s*s) and subtract gradient of denominator
        ds = np.ones_like(h)
        soft_max_val = m + np.log(sum_exp) / beta
        return 2.0 * N * (grad / (s * s) - 2.0 * soft_max_val * ds / (s * s * s))

def _cosine_param_objective(params):
    """
    Objective for cosine model:
    h[i] = baseline + amp * cos(2*pi*freq*i/N + phase)
    params = [baseline, amp, freq, phase]
    """
    baseline, amp, phase, freq = params
    i = np.arange(N)
    h = baseline + amp * np.cos(2 * np.pi * freq * i / N + phase)
    return _soft_max_obj(h, beta=100.0)

def _dct_optimize(h0, max_iter=100):
    """Optimize using DCT parameterization."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _dct_objective(coeffs):
        h = _dct_to_h(coeffs)
        return _soft_max_obj(h, beta=50.0)
    
    try:
        res = minimize(_dct_objective, dct_coeffs.copy(),
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _dct_to_h(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _low_freq_spectral_optimize(h0, num_modes=20, max_iter=100):
    """Optimize only low-frequency DCT modes."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    low_coeffs = dct_coeffs[:num_modes].copy()
    high_coeffs = dct_coeffs[num_modes:].copy()
    
    def _reconstruct_low(low_coeffs_vec):
        coeffs = np.concatenate([low_coeffs_vec, high_coeffs])
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _low_mode_objective(low_coeffs_vec):
        h = _reconstruct_low(low_coeffs_vec)
        return _soft_max_obj(h, beta=50.0)
    
    try:
        res = minimize(_low_mode_objective, low_coeffs.copy(),
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _reconstruct_low(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Standard diverse initializations
    # Random normal
    for _ in range(3):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        candidates.append(h0)
    
    # Comb patterns
    for period in [2, 3, 4]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        candidates.append(h0)
        
    # Waves
    for freq in [2, 5, 10]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        candidates.append(h0)
        
    # Block patterns
    for block_size in [10, 20]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        candidates.append(h0)
        
    # 2. Cosine-based initializations
    for freq in [10, 20, 50]:
        i = np.arange(N)
        h0 = 1.0 + 0.5 * np.cos(2 * np.pi * freq * i / N)
        candidates.append(h0)

    # Process candidates with time-aware optimization
    for idx, h0 in enumerate(candidates):
        if time.time() > deadline - 35.0:
            break
        
        h_curr = h0.copy()
        
        # Step 1: DCT optimization
        if time.time() < deadline - 25.0:
            h_dct = _dct_optimize(h_curr, max_iter=40)
            val_dct = _objective_hard(h_dct)
            if val_dct < _objective_hard(h_curr):
                h_curr = h_dct
        
        # Step 2: Low-freq spectral optimization
        if time.time() < deadline - 15.0:
            h_low = _low_freq_spectral_optimize(h_curr, num_modes=15, max_iter=40)
            val_low = _objective_hard(h_low)
            if val_low < _objective_hard(h_curr):
                h_curr = h_low
        
        # Step 3: Direct L-BFGS-B on h
        if time.time() < deadline - 10.0:
            try:
                res = minimize(_soft_max_obj, h_curr,
                              method='L-BFGS-B',
                              args=(50.0,),
                              options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6})
                val = _objective_hard(res.x)
                if val < _objective_hard(h_curr):
                    h_curr = res.x
            except Exception:
                pass
        
        val_curr = _objective_hard(h_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()

    # Cosine Parameter Optimization
    if time.time() < deadline - 15.0:
        for _ in range(15):
            if time.time() > deadline - 8.0:
                break
            baseline = rng.uniform(-1, 1)
            amp = rng.uniform(0.1, 2.0)
            phase = rng.uniform(0, 2*np.pi)
            freq = rng.uniform(1, N/2)
            
            init_params = np.array([baseline, amp, phase, freq])
            
            try:
                res = minimize(_cosine_param_objective, init_params,
                              method='Nelder-Mead',
                              options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.fun < 1e9:
                    baseline, amp, phase, freq = res.x
                    i = np.arange(N)
                    h_cos = baseline + amp * np.cos(2 * np.pi * freq * i / N + phase)
                    val_cos = _objective_hard(h_cos)
                    if val_cos < best_val:
                        best_val = val_cos
                        best_h = h_cos.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < deadline - 3.0:
        for beta in [100.0, 200.0]:
            if time.time() > deadline - 1.0:
                break
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
