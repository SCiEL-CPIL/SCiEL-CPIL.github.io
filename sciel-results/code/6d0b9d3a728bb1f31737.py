import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    dx = 1.0 / (2.0 * N)
    
    # Initial guess: raised cosine
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx / 2
    f = 1.0 + 0.5 * np.cos(2 * np.pi * x)
    f = np.maximum(f, 0.0)
    integral = np.sum(f) * dx
    f = f / integral
    
    best_f = f.copy()
    best_peak = _compute_peak_linear(f, dx)
    
    # Adam optimizer
    lr = 0.005
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    
    for step in range(50000):
        if time.time() > deadline - 3.0:
            break
        
        t += 1
        integral = np.sum(f) * dx
        
        # Compute linear autoconvolution using zero-padded FFT
        fft_len = 2 * N
        fft_f = np.fft.fft(f, n=fft_len)
        autoconv_freq = np.abs(fft_f) ** 2
        autoconv_full = np.real(np.fft.ifft(autoconv_freq)) * dx  # scale by dx for integral
        # The autoconvolution at the center (x=0) is autoconv_full[N]
        # Actually, the linear convolution of two length-N sequences has length 2N-1
        # The peak is typically at the center
        
        # Find the peak of the autoconvolution
        # The autoconvolution is symmetric, peak at center
        peak_idx = N  # center of the 2N-1 length result (approximately)
        # Actually let's find the true max
        peak_val = np.max(autoconv_full)
        peak_idx = np.argmax(autoconv_full)
        
        norm = integral ** 2
        normalized_peak = peak_val / norm if norm > 1e-15 else np.inf
        
        if normalized_peak < best_peak:
            best_peak = normalized_peak
            best_f = f.copy()
        
        # Gradient: d(peak_val)/df[i]
        # autoconv_full[k] = sum_j f[j] * f[j + k - (N-1)] * dx for valid indices
        # This is messy. Let's use: autoconv_full = ifft(|fft(f, n=2N)|^2) * dx
        # d(autoconv_full[peak_idx])/df[i] = 2 * f[peak_idx - i + offset] * dx (approximately)
        
        # For linear convolution: (f*f)[k] = sum_{j} f[j] f[k-j]
        # d(f*f)[k]/df[i] = f[k-i] + f[k-i] = 2*f[k-i] (with appropriate bounds)
        
        # In the zero-padded FFT representation, the relationship is more complex.
        # Let me use a numerical approach or a simpler analytical gradient.
        
        # Simpler gradient: for the peak at position p in the linear convolution
        # dC[p]/df[i] = 2 * f[i] * f[p - i] type relationship
        # But this depends on the exact indexing.
        
        # Let me use: the autoconvolution peak is approximately at x=0
        # C(0) = sum_j f[j]^2 * dx (if we align properly)
        # Actually C(0) = ∫ f(t) f(-t) dt = ∫ f(t)^2 dt for symmetric f
        # But f is not necessarily symmetric.
        
        # Let me just use a subgradient: the peak is max over k of C(k)
        # d(max_k C(k))/df[i] = dC[k*]/df[i] where k* is argmax
        # C(k) = sum_j f[j] f[k-j] (linear convolution, with f=0 outside [0,N-1])
        # dC(k)/df[i] = f[k-i] + f[k-i] = 2*f[k-i] if 0<=k-i<N, else f[k-i] if k-i==i... 
        # More carefully: dC(k)/df[i] = f[k-i] (from j=i term) + f[k-i] (from j=k-i term) 
        # = 2*f[k-i] if i != k-i, i.e., 2i != k
        # = f[k-i] if 2i == k
        
        # For the zero-padded FFT, the peak index in the full array corresponds to
        # the linear convolution index. Let me map it.
        
        # The linear convolution index p corresponds to autoconv_full[p + (N-1)] approximately
        # This is getting complicated. Let me use a different strategy.
        
        # Alternative: use the fact that for nonnegative f, the peak of f*f 
        # is bounded and we can use coordinate descent or a simpler method.
        
        # Let me try: minimize sum of squares (L2 norm of autoconvolution) as a proxy,
        # which is easier to differentiate.
        # ||f*f||_2^2 = ||FFT(f*f)||_2^2 = |||FFT(f)|^2||_2^2 = sum_m |FFT(f)[m]|^4
        # This is smooth and differentiable.
        
        # d(sum_m |F[m]|^4)/df[i] = 4 * sum_m |F[m]|^2 * Re(conj(F[m]) * exp(-2*pi*i*m/N))
        # where F[m] = FFT(f)[m]
        
        F = np.fft.fft(f, n=fft_len)
        absF_sq = np.abs(F) ** 2
        
        # Gradient of sum |F|^4 w.r.t. f[i]
        # = 4 * sum_m |F[m]|^2 * Re(conj(F[m]) * W[i,m])
        # where W[i,m] = exp(-2*pi*i*m/N) for the DFT
        
        # This is: 4 * Re( sum_m |F[m]|^2 * conj(F[m]) * exp(-2*pi*i*m/N) )
        # = 4 * Re( conj( sum_m |F[m]|^2 * F[m] * exp(2*pi*i*m/N) ) )
        
        # Let G[m] = |F[m]|^2 * F[m] = F[m]^3 (in magnitude sense, but with phase)
        # Actually G[m] = |F[m]|^2 * F[m]
        # Then grad[i] = 4 * Re( sum_m G[m] * exp(-2*pi*i*m/N) )
        # = 4 * Re( IFFT(G) [i] ) * (N/fft_len) for the scaling
        
        G = absF_sq * F
        grad_l2 = 4.0 * np.real(np.fft.ifft(G, n=fft_len)[:N])
        
        # Scale for the zero-padded FFT
        grad_l2 = grad_l2 * (fft_len / N)  # FFT scaling
        
        # Also account for normalization by integral^2
        # d(normalized_peak)/df[i] ≈ d(L2_proxy)/df[i] / norm - L2_proxy * d(norm)/df[i] / norm^2
        # But we're using L2 as proxy, not the actual peak. Let me just minimize L2.
        
        # For the actual normalized quantity, we need to divide by integral^2
        # d(L2/integral^2)/df[i] = (grad_l2[i] * integral^2 - L2 * 2*integral*dx) / integral^4
        # = grad_l2[i] / integral^2 - 2*dx*L2 / integral^3
        
        L2_val = np.sum(absF_sq ** 2)  # This is sum |F|^4
        # Actually the L2 norm of autoconvolution squared is (1/N) * sum |F|^4 for DFT
        # Let me just use the raw sum
        
        grad = grad_l2 / (integral ** 2) - 2.0 * dx * L2_val / (integral ** 3)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        f_new = f - update
        f_new = np.maximum(f_new, 0.0)
        
        new_int = np.sum(f_new) * dx
        if new_int > 1e-12:
            f_new = f_new * (integral / new_int)
        
        f = f_new
    
    return list(best_f)


def _compute_peak_linear(f, dx):
    """Compute normalized peak of linear autoconvolution."""
    fft_len = 2 * N
    fft_f = np.fft.fft(f, n=fft_len)
    autoconv_freq = np.abs(fft_f) ** 2
    autoconv_full = np.real(np.fft.ifft(autoconv_freq)) * dx
    integral = np.sum(f) * dx
    norm = integral ** 2
    if norm < 1e-15:
        return np.inf
    return np.max(autoconv_full) / norm
