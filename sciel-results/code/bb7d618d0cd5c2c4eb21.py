import numpy as np
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

# Precompute FFT setup
# The step function has N intervals on [-1/4, 1/4], total width 0.5
K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    # Normalized score: max_conv / (integral_f)^2
    score = max_conv / (integral_f ** 2)
    
    return score

def _autoconv_and_grad(heights):
    """
    Compute autoconvolution, its peak value, peak index, and gradient of peak w.r.t. heights.
    Uses soft-max smoothing for differentiability.
    """
    heights = np.asarray(heights, dtype=np.float64)
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Soft-max with temperature for smooth max
    tau = 0.01
    conv_shifted = conv / tau
    max_conv_shifted = np.max(conv_shifted)
    softmax_weights = np.exp(conv_shifted - max_conv_shifted)
    softmax_weights /= np.sum(softmax_weights)
    
    soft_max = np.sum(softmax_weights * conv)
    
    # Gradient of soft-max w.r.t. f_fine
    # d(soft_max)/d(conv[k]) = softmax_weights[k]
    # d(conv[k])/d(f_fine[j]) = f_fine[(k-j) % M_FFT] (circular)
    # So d(soft_max)/d(f_fine[j]) = sum_k softmax_weights[k] * f_fine[(k-j) % M_FFT]
    
    # Efficient: this is a circular cross-correlation of softmax_weights with f_fine
    # d(soft_max)/d(f_fine) = ifft(fft(softmax_weights) * conj(fft(f_fine))) ... 
    # Actually, let's think again.
    # conv[k] = sum_j f[j] * f[(k-j) % M]
    # d(conv[k])/d(f[j]) = f[(k-j) % M]
    # d(soft_max)/d(f[j]) = sum_k w_k * f[(k-j) % M]
    # Let g[j] = sum_k w_k * f[(k-j) % M]
    # This is the circular cross-correlation of w and f.
    # g = ifft(fft(w) * conj(fft(f)))
    # Wait, cross-correlation of a and b is ifft(fft(a) * conj(fft(b)))
    # We want sum_k w_k * f[(k-j) % M]
    # Let's define h[j] = sum_k w_k * f[(k-j) % M]
    # This is the circular convolution of w_rev with f, where w_rev[k] = w[-k % M]
    # Or equivalently, ifft(fft(w) * conj(fft(f))) gives correlation?
    # Let's just compute it via FFT.
    # g[j] = sum_k w_k * f[(k-j) % M]
    # Let m = k - j, so k = m + j
    # g[j] = sum_m w[m+j] * f[m]
    # This is the circular cross-correlation of w and f.
    # In frequency domain: G = W * conj(F)
    # g = ifft(W * conj(F))
    
    W = np.fft.rfft(softmax_weights)
    G = np.fft.irfft(W * np.conj(F), n=M_FFT)
    
    # G[j] = d(soft_max)/d(f_fine[j])
    # Now map to heights
    # f_fine[j] = h_{j//K} for j in [0, M_STEP)
    # d(soft_max)/d(h_i) = sum_{j: j//K == i, j < M_STEP} G[j]
    
    G_step = G[:M_STEP]
    G_reshaped = G_step.reshape(N, K)
    grad_conv = np.sum(G_reshaped, axis=1)
    
    integral_f = np.sum(heights) * w
    score = soft_max / (integral_f ** 2)
    
    # Gradient of score w.r.t. heights
    # score = soft_max / I^2
    # d(score)/d(h_i) = (grad_conv[i] * I^2 - soft_max * 2*I * w) / I^4
    # = grad_conv[i] / I^2 - 2 * soft_max * w / I^3
    
    grad_score = grad_conv / (integral_f ** 2) - 2 * soft_max * w / (integral_f ** 3)
    
    return score, grad_score

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Strategy 1: Try various smooth shapes as initializations
    x = np.linspace(0, 1, N)
    
    # Raised cosine variations
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
        h_cos = np.maximum(h_cos, 0.0)
        score = compute_score(h_cos)
        if score < best_score:
            best_score = score
            best_h = h_cos.copy()
    
    # Plateau with tapered edges
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        h_plateau = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 0:
            ramp = np.linspace(0, 1, taper_len)
            h_plateau[:taper_len] = ramp
            h_plateau[-taper_len:] = ramp[::-1]
        score = compute_score(h_plateau)
        if score < best_score:
            best_score = score
            best_h = h_plateau.copy()
    
    # Gaussian-like shapes
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        h_gauss = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        h_gauss = h_gauss / np.max(h_gauss)
        score = compute_score(h_gauss)
        if score < best_score:
            best_score = score
            best_h = h_gauss.copy()
    
    # Strategy 2: Random smooth perturbations
    rng = np.random.default_rng(42)
    for trial in range(50):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        # Create smooth perturbation using low-frequency Fourier modes
        g = np.zeros(N)
        num_modes = rng.integers(2, 10)
        for _ in range(num_modes):
            k = rng.integers(1, N // 4)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.05, 0.3)
            g += amp * np.cos(2 * np.pi * k * x + phase)
        
        g -= np.mean(g)
        
        for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            h_trial = 1.0 + eps * g
            h_trial = np.maximum(h_trial, 0.0)
            if np.sum(h_trial) <= 0:
                continue
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # Strategy 3: Projected gradient descent with soft-max smoothing
    if time.time() - START_TIME < TIME_LIMIT - 20:
        h_ref = best_h.copy()
        lr = 0.002
        for step in range(2000):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            score, grad = _autoconv_and_grad(h_ref)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            h_new = h_ref - lr * grad
            h_new = np.maximum(h_new, 0.0)
            
            # Adaptive learning rate with backtracking
            new_score = compute_score(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
                h_ref = h_new.copy()
                lr *= 1.02
            else:
                # Backtrack
                lr *= 0.95
                h_ref = best_h.copy()
            
            # Check for convergence
            if step > 100 and abs(score - best_score) < 1e-10:
                break
    
    # Strategy 4: Final local search
    if time.time() - START_TIME < TIME_LIMIT - 5:
        # Try scaling the best shape
        for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            h_scaled = best_h * scale
            score = compute_score(h_scaled)
            if score < best_score:
                best_score = score
                best_h = h_scaled.copy()
        
        # Try local search around best
        rng2 = np.random.default_rng(123)
        for _ in range(100):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng2.normal(0, 0.005, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # Final gradient polish
    if time.time() - START_TIME < TIME_LIMIT - 2:
        h_final = best_h.copy()
        lr = 0.0005
        for step in range(500):
            if time.time() - START_TIME > TIME_LIMIT - 1:
                break
            
            score, grad = _autoconv_and_grad(h_final)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            h_final = h_final - lr * grad
            h_final = np.maximum(h_final, 0.0)
            
            new_score = compute_score(h_final)
            if new_score < best_score:
                best_score = new_score
                best_h = h_final.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
