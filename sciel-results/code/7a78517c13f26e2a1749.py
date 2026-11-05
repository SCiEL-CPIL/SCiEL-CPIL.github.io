"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT (O(N log N) instead of O(N^2))."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2*n-1]

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

def _soft_max_obj_grad(x, beta=50.0):
    """Objective and gradient for soft-max objective."""
    h = x
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
        # d(soft_max)/d(conv[i]) = exp(beta*(conv[i]-m)) / sum_exp
        grad_conv = exp_shifted / sum_exp
        
        # Gradient of conv w.r.t. h
        # conv = h * h (full), so d(conv[k])/d(h[i]) = h[k-i] + h[k-i] if valid
        # More precisely, conv[k] = sum_i h[i]*h[k-i]
        # d(conv[k])/d(h[i]) = h[k-i] if 0 <= k-i < N, plus h[i-k] if 0 <= i-k < N
        # Actually for full convolution: conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[k-i]
        # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, plus h[j-k] if 0<=j-k<N
        # But since we're using FFT, let's compute gradient via FFT
        
        # grad_conv is a vector of length 2N-1
        # d(conv[k])/d(h[j]) for k=0..2N-2
        # conv[k] = sum_{i} h[i] h[k-i] where 0<=i<N and 0<=k-i<N
        # So d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, plus h[j-k] if 0<=j-k<N
        # But h[k-j] and h[j-k] are the same when we sum over all valid pairs
        
        # Actually, let's think more carefully:
        # conv = h * h (linear convolution)
        # d(conv)/d(h[j]) is the convolution of delta_j with h, plus h with delta_j
        # = h shifted by j, plus h shifted by (N-1-j) reflected... 
        # Actually for linear convolution: conv[k] = sum_i h[i] h[k-i]
        # d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N, i.e., k<j)
        # Wait, no. If we differentiate conv[k] = sum_i h[i] h[k-i] w.r.t. h[j]:
        # = h[k-j] (from the h[i] term when i=j) + h[j-k] (from the h[k-i] term when k-i=j, i.e., i=k-j)
        # But h[k-j] is only valid if 0<=k-j<N, and h[j-k] is only valid if 0<=j-k<N
        # These are the same condition essentially: one is valid when k>=j, the other when k<j
        # Actually: h[k-j] is the contribution when i=j, valid if 0<=k-j<N i.e. j<=k<N+j
        # h[j-k] is the contribution when k-i=j i.e. i=k-j, valid if 0<=k-j<N i.e. j<=k<N+j
        # Wait, that's the same condition! So actually d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] if j<=k<N+j
        # Hmm, but that's not right either because the two terms are from different summation indices.
        
        # Let me just compute it directly using the structure:
        # grad_h[j] = sum_k grad_conv[k] * d(conv[k])/d(h[j])
        # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N else 0, PLUS h[j-k] if 0<=j-k<N else 0
        # But h[j-k] = h[(j-k)] which is valid if 0<=j-k<N, i.e., k<=j and k>=j-N+1
        # And h[k-j] is valid if 0<=k-j<N, i.e., k>=j and k<N+j
        
        # Actually, I realize the convolution is symmetric: h*h = h*h, so the gradient should be
        # grad_h = 2 * (grad_conv * h) where * denotes appropriate correlation
        
        # Let's use the fact that for linear convolution c = a*b:
        # d(c[k])/d(a[j]) = b[k-j] if 0<=k-j<len(b)
        # d(c[k])/d(b[j]) = a[k-j] if 0<=k-j<len(a)
        # Here a=b=h, so:
        # d(c[k])/d(h[j]) = h[k-j] (from a term) + h[k-j] (from b term, since a=b)
        # Wait, no. c[k] = sum_i h[i] h[k-i]
        # d(c[k])/d(h[j]) = h[k-j] (when i=j) + sum over i where k-i=j, i.e., i=k-j: h[k-j]
        # But the second term only exists if 0<=k-j<N
        # So d(c[k])/d(h[j]) = h[k-j] if 0<=k-j<N, plus h[k-j] if 0<=k-j<N
        # = 2*h[k-j] if 0<=k-j<N, else 0
        # Hmm, but that double-counts when i=j and k-i=j, which happens when k=2j
        # Actually no, the two terms are distinct: one is from the first h, one from the second h
        # So yes, d(c[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N
        
        # So grad_h[j] = sum_k grad_conv[k] * 2*h[k-j] for k where 0<=k-j<N
        # = 2 * sum_{k=j}^{N+j-1} grad_conv[k] * h[k-j]
        # = 2 * (grad_conv * h)[j] where * is cross-correlation
        
        # Let's compute this via FFT:
        # grad_h = 2 * correlate(grad_conv, h)
        # correlate(a, b)[j] = sum_k a[k+j] * b[k]
        
        n_conv = len(conv)
        n_fft2 = 1
        while n_fft2 < n_conv + N - 1:
            n_fft2 *= 2
        GC = np.fft.rfft(grad_conv, n=n_fft2)
        HH = np.fft.rfft(h, n=n_fft2)
        # Cross-correlation: correlate(grad_conv, h)
        # In FFT: ifft(fft(grad_conv) * conj(fft(h)))
        corr = np.fft.irfft(GC * np.conj(HH), n=n_fft2)[:n_conv]
        # Actually, let me just do it directly for correctness
        grad_h = np.zeros(N)
        for j in range(N):
            # grad_h[j] = 2 * sum_{k=0}^{N-1} grad_conv[j+k] * h[k] if j+k < n_conv
            for k in range(N):
                idx = j + k
                if idx < n_conv:
                    grad_h[j] += 2.0 * grad_conv[idx] * h[k]
        
        val = 2.0 * N * soft_max / (s * s)
        # d(val)/d(h[j]) = 2*N/(s^2) * d(soft_max)/d(h[j]) - 2*N*soft_max/s^3 * 2*s * d(s)/d(h[j])
        # d(s)/d(h[j]) = 1
        # d(soft_max)/d(h[j]) = grad_h[j]
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * soft_max / (s * s * s)
        return val, grad_val
    else:
        val = 2.0 * N * m / (s * s)
        # Gradient is just from the max term, which is non-smooth
        # Approximate by picking the argmax
        idx_max = np.argmax(conv)
        # d(conv[idx_max])/d(h[j]) = 2*h[idx_max-j] if 0<=idx_max-j<N
        grad_h = np.zeros(N)
        for j in range(N):
            d = idx_max - j
            if 0 <= d < N:
                grad_h[j] = 2.0 * h[d]
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * m / (s * s * s)
        return val, grad_val

def _dct_optimize(h0, deadline, rng, max_iter=200):
    """Optimize using DCT parameterization with spectral amplitude control."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _dct_objective(coeffs):
        h = _dct_to_h(coeffs)
        return _soft_max_obj(h)
    
    coeff_init = dct_coeffs.copy()
    
    try:
        res = minimize(_dct_objective, coeff_init,
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _dct_to_h(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _spectral_shape_optimize(h0, deadline, rng, max_iter=100):
    """Optimize spectral shape by controlling DCT amplitude spectrum."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    amplitudes = np.abs(dct_coeffs)
    signs = np.sign(dct_coeffs)
    
    log_amps = np.log(amplitudes + 1e-10)
    
    def _reconstruct_from_log_amps(log_amps_vec):
        amps = np.exp(log_amps_vec)
        coeffs = signs * amps
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _log_amp_objective(log_amps_vec):
        h = _reconstruct_from_log_amps(log_amps_vec)
        return _soft_max_obj(h)
    
    try:
        res = minimize(_log_amp_objective, log_amps.copy(),
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _reconstruct_from_log_amps(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _low_freq_spectral_optimize(h0, deadline, rng, num_modes=20):
    """Optimize only low-frequency DCT modes for better spectral shape."""
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
        return _soft_max_obj(h)
    
    try:
        res = minimize(_low_mode_objective, low_coeffs.copy(),
                      method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _reconstruct_low(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _random_walk_init(rng, N):
    """Generate random walk with negative drift as initialization."""
    drift = -0.01
    noise_scale = 0.5
    h = np.zeros(N)
    for i in range(N):
        h[i] = h[i-1] + drift + noise_scale * rng.standard_normal() if i > 0 else noise_scale * rng.standard_normal()
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
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
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 6. DCT-shaped initializations
    for num_modes in [5, 10, 20]:
        h0 = np.zeros(N)
        coeffs = np.zeros(N)
        for k in range(num_modes):
            coeffs[k] = rng.standard_normal() * (1.0 / (k + 1))
        zm = idct(coeffs, type=3, norm='ortho')
        h0 = 1.0 + zm
        initializations.append(h0)
    
    # 7. Random walk initializations (from knowledge: improved 1.245 -> 1.313)
    for _ in range(5):
        h0 = _random_walk_init(rng, N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # Optimize each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        
        # Try DCT optimization
        if time.time() < deadline - 15.0:
            h_dct = _dct_optimize(h_curr, deadline, rng, max_iter=30)
            val_dct = _objective_hard(h_dct)
            if val_dct < _objective_hard(h_curr):
                h_curr = h_dct
        
        # Try spectral shape optimization
        if time.time() < deadline - 12.0:
            h_spec = _spectral_shape_optimize(h_curr, deadline, rng, max_iter=30)
            val_spec = _objective_hard(h_spec)
            if val_spec < _objective_hard(h_curr):
                h_curr = h_spec
        
        # Try low-frequency spectral optimization
        if time.time() < deadline - 10.0:
            h_low = _low_freq_spectral_optimize(h_curr, deadline, rng, num_modes=15)
            val_low = _objective_hard(h_low)
            if val_low < _objective_hard(h_curr):
                h_curr = h_low
        
        # Update best
        val_curr = _objective_hard(h_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0, 500.0]:
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Final direct optimization with high beta
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_soft_max_obj, best_h,
                          method='L-BFGS-B',
                          args=(1000.0,),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
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
