import numpy as np
import time

N = 600
DELTA = 1.0 / N
L = 0.5
_NFFT = 1 << (2 * N - 1).bit_length()

def compute_ratio_fast(h):
    """Compute max(f*f) / (int f)^2 for step function with heights h, using FFT."""
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0)
    s = np.sum(h) * DELTA
    if s < 1e-15:
        return 1e15
    h_norm = h / s
    F = np.fft.rfft(h_norm, _NFFT)
    conv = np.fft.irfft(F * F, _NFFT)[:2*N-1]
    max_val = np.max(conv) * DELTA
    return max_val

def soft_max(x, tau):
    """Compute tau * log(sum(exp(x/tau)))"""
    # Shift for numerical stability
    x_max = np.max(x)
    if x_max > 1000 * tau:
        # Avoid overflow
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))
    return tau * np.log(np.sum(np.exp(x / tau)))

def compute_grad_ratio(h, tau):
    """
    Compute gradient of soft_max(conv) w.r.t h.
    h is normalized such that sum(h)*DELTA = 1.
    """
    s = np.sum(h) * DELTA
    h_norm = h / s
    
    F = np.fft.rfft(h_norm, _NFFT)
    conv = np.fft.irfft(F * F, _NFFT)[:2*N-1]
    
    # Gradient of soft_max w.r.t conv
    exp_conv = np.exp((conv - np.max(conv)) / tau)
    sum_exp = np.sum(exp_conv)
    grad_conv = exp_conv / sum_exp
    
    # Gradient of conv w.r.t h_norm
    # conv[k] = sum_i h_norm[i] h_norm[i-k]
    # d conv[k] / d h_norm[j] = h_norm[j-k] + h_norm[j+k]
    # This is the convolution of h_norm with h_norm, but with a specific structure.
    # Actually, d (h*h)[k] / d h[j] = h[j-k] + h[j+k] (with wrap-around or linear?)
    # For linear convolution, it's the sum of h shifted by k.
    # Let's compute it via FFT.
    # The operation C = h * h (linear).
    # Gradient G = d C / d h.
    # C is circular if we use FFT without zero-padding? No, we used _NFFT.
    # The linear convolution is embedded in the circular convolution of size _NFFT.
    # The gradient of the linear convolution w.r.t h is the "cross-correlation" with itself?
    # Actually, C_k = sum_i h_i h_{i-k}.
    # d C_k / d h_j = h_{j-k} + h_{j+k} (if indices valid).
    # This is equivalent to convolving h with h, but taking the "sum of shifts" part.
    # Note that h*h = C.
    # The gradient of the sum of squares? No.
    # Let's use the property that d (h*h) / d h = 2 * (h * h) is wrong.
    # d (h*h)[k] / d h[j] = h[j-k] + h[j+k].
    # Let's compute this vector G of length N.
    # G[j] = sum_k grad_conv[k] * (h[j-k] + h[j+k])
    # This can be done by convolving grad_conv with h.
    
    # We need to map conv (length 2N-1) back to the gradient on h (length N).
    # conv indices: 0 to 2N-2.
    # h indices: 0 to N-1.
    # Shift k corresponds to conv index N-1+k.
    # So grad_conv_full of length 2N-1.
    # We need to compute G = IFFT( FFT(h) * FFT(gradient_of_conv_structure) )?
    
    # Simpler:
    # G[j] = sum_k grad_conv[k] * h[j-k] + sum_k grad_conv[k] * h[j+k]
    # Let k' = k - (N-1). Then k ranges from -(N-1) to N-1.
    # conv[k] in our array is at index k + N - 1.
    # So grad_conv_arr[m] corresponds to shift k = m - (N-1).
    
    # Term 1: sum_k grad_conv[k] h[j-k]
    # This is the convolution of grad_conv (centered) with h.
    # Let's create a zero-padded grad_conv of size _NFFT.
    # Place grad_conv[m] at index (m + (N-1)) in the FFT domain?
    # No, let's just use FFT for the convolution.
    
    # grad_conv is defined for lags k = -(N-1) ... N-1.
    # Let's create an array G_conv of size _NFFT.
    # G_conv[N-1 + k] = grad_conv[k] for k in 0..N-1
    # G_conv[N-1 - k] = grad_conv[k] for k in 1..N-1
    # Wait, the lag k in h[j-k] means we shift h by k.
    
    # Let's compute G1[j] = sum_k grad_conv[k] h[j-k].
    # This is the linear convolution of h and grad_conv (reversed).
    # Let g[k] = grad_conv[k] for k=0..N-1 and g[k] = grad_conv[2N-2-k] for k=N..2N-2?
    # No, the lags are symmetric.
    
    # Let's just do it directly with FFT.
    # G1 = ifft(fft(h) * fft(grad_conv_padded))
    # Where grad_conv_padded is arranged such that its FFT corresponds to the kernel.
    
    # Kernel K such that G1[j] = sum_k K[k] h[j-k].
    # K[k] = grad_conv[k] for k >= 0?
    # conv index m corresponds to lag k = m - (N-1).
    # So grad_conv[m] is the weight for lag k = m - N + 1.
    # We want G1[j] = sum_m grad_conv[m] h[j - (m - N + 1)].
    # Let k = m - N + 1. m = k + N - 1.
    # G1[j] = sum_k grad_conv[k+N-1] h[j-k].
    # This is the convolution of h with the sequence v[k] = grad_conv[k+N-1].
    # v is of length 2N-1.
    
    # Similarly for G2[j] = sum_k grad_conv[k] h[j+k].
    # G2[j] = sum_k grad_conv[k+N-1] h[j+k].
    # This is the cross-correlation.
    # G2[j] = sum_k v[k] h[j+k].
    # Note that sum_k v[k] h[j+k] = sum_k v[-k] h[j-k].
    # So G2 is convolution of h with v_reversed.
    
    # So Gradient G = G1 + G2 = conv(h, v) + conv(h, v_reversed).
    
    v = conv.copy() # Length 2N-1
    # v[m] corresponds to lag k = m - N + 1.
    # We need to embed v into an FFT array of size _NFFT.
    # The convolution is linear.
    
    # Pad v to _NFFT.
    # v is centered? No, v[0] is lag -(N-1). v[N-1] is lag 0.
    # In FFT convolution, the kernel is usually centered at 0 or start at 0.
    # If we use `np.fft.ifft(np.fft.fft(h) * np.fft.fft(kernel))`, the kernel is assumed to be circularly shifted.
    # Let's place v in the kernel array.
    # kernel[0:N-1] = v[N-1:2N-1] (lags 0 to N-1)
    # kernel[N-1:_NFFT] = v[0:N-1] (lags -(N-1) to -1)
    # This assumes the kernel is circularly aligned with the FFT output.
    
    kernel = np.zeros(_NFFT, dtype=np.float64)
    # v index m -> lag k = m - N + 1.
    # We want kernel[k] to hold the weight for lag k.
    # In FFT, if we multiply FFT(h) by FFT(kernel), the result is circular convolution.
    # For linear convolution, we need to handle the wrap-around.
    # Since _NFFT >= 2N-1, there is no overlap if we place correctly.
    
    # Place v such that lag 0 is at index 0?
    # If kernel[0] = v[N-1] (lag 0)
    # kernel[1] = v[N] (lag 1)
    # ...
    # kernel[N-1] = v[2N-2] (lag N-1)
    # kernel[_NFFT - 1] = v[0] (lag -(N-1))
    # ...
    # kernel[_NFFT - (N-1)] = v[N-2] (lag -1)
    
    kernel[0:N] = v[N-1:2*N-1]
    kernel[_NFFT - (N-1):_NFFT] = v[0:N-1]
    
    G1 = np.fft.irfft(np.fft.rfft(h_norm, _NFFT) * np.fft.rfft(kernel, _NFFT), _NFFT)[:N]
    
    # For G2, we need v_reversed.
    # v_rev[k] = v[-k].
    # Lag k in G2 uses v[k].
    # In the convolution form G2[j] = sum_k v[-k] h[j-k].
    # So kernel2[k] = v[-k].
    # v[-k] = v[ (N-1) - k ]? No.
    # v[m] is weight for lag k = m - N + 1.
    # So weight for lag -k is v[m] where m - N + 1 = -k => m = N - 1 - k.
    # So kernel2[lag k] = v[N - 1 - k].
    
    kernel2 = np.zeros(_NFFT, dtype=np.float64)
    # kernel2[0] (lag 0) = v[N-1]
    # kernel2[1] (lag 1) = v[N-2]
    # ...
    # kernel2[N-1] (lag N-1) = v[0]
    # kernel2[_NFFT-1] (lag -(N-1)) = v[2N-2]
    
    kernel2[0:N] = v[N-1:-1:-1]
    kernel2[_NFFT - (N-1):_NFFT] = v[2*N-2:N-1:-1]
    
    G2 = np.fft.irfft(np.fft.rfft(h_norm, _NFFT) * np.fft.rfft(kernel2, _NFFT), _NFFT)[:N]
    
    grad_h = G1 + G2
    
    # Chain rule: h_norm = h / s.
    # J(h) = soft_max(conv(h/s)).
    # dJ/dh = (1/s) * dJ/dh_norm - (h/s^2) * sum(h_norm * dJ/dh_norm)?
    # Actually, we can just compute gradient w.r.t h_norm and scale.
    # But we need to project back to h.
    # Let's return gradient w.r.t h_norm.
    
    return grad_h * DELTA, max_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    x = np.linspace(-0.25, 0.25, N+1)[:-1] + DELTA/2
    
    # Initialize with a good candidate
    # Uniform
    h = np.ones(N)
    
    # Try to improve using gradient descent with soft-max
    # We optimize h directly.
    
    # Normalization
    s = np.sum(h) * DELTA
    h = h / s
    
    tau = 0.01
    alpha = 1e-3
    
    best_h = h.copy()
    best_val = compute_ratio_fast(h)
    
    # Gradient Descent
    for i_iter in range(2000):
        if time.time() - start_time > deadline - 5:
            break
        
        grad, val = compute_grad_ratio(h, tau)
        
        # Gradient step
        h_new = h - alpha * grad
        
        # Project to simplex: sum(h)*DELTA = 1, h >= 0
        h_new = np.maximum(h_new, 0)
        s_new = np.sum(h_new) * DELTA
        if s_new > 1e-15:
            h_new = h_new / s_new
        else:
            h_new = h.copy()
            
        val_new = compute_ratio_fast(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new
            # Adaptive alpha
            # alpha *= 1.1
        else:
            # Backtrack
            alpha *= 0.5
            if alpha < 1e-8:
                # Try changing tau
                tau *= 0.5
                alpha = 1e-3
                if tau < 1e-4:
                    break
                    
    # Final polish with coordinate descent if time permits
    if time.time() - start_time < deadline - 10:
        h = best_h
        for i_iter in range(500):
            if time.time() - start_time > deadline - 2:
                break
            improved = False
            order = np.random.permutation(N)
            for i in order:
                if time.time() - start_time > deadline - 2:
                    break
                old_val = h[i]
                for frac in [0.5, 0.8, 1.2, 1.5, 0.0]:
                    new_val = old_val * frac
                    h_test = h.copy()
                    h_test[i] = new_val
                    s_test = np.sum(h_test) * DELTA
                    if s_test < 1e-15: continue
                    h_test = h_test / s_test
                    r_test = compute_ratio_fast(h_test)
                    if r_test < best_val - 1e-14:
                        best_val = r_test
                        best_h = h_test
                        h = h_test
                        improved = True
                        break
            if not improved:
                break

    h_final = best_h
    s_final = np.sum(h_final) * DELTA
    if s_final > 1e-15:
        h_final = h_final / s_final
        
    return [float(x) for x in h_final]
