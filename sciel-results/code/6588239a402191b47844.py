import numpy as np
import time
from scipy.optimize import minimize

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

def _spectrum_to_heights(spectral_params):
    """
    Convert spectral parameters to time-domain heights.
    
    Parameters:
    - spectral_params: array of shape (n_params,)
      We use a parametric power spectrum: P(k) = a0 + a1*exp(-k^2/sigma^2) + a2*cos(2*pi*k*N/4 + phase)
      Actually, let's use a simpler approach:
      P(k) = base + amp * exp(-((k - center)^2) / (2*sigma^2))
      
    We parameterize as:
    - params[0]: base level (DC component)
    - params[1]: amplitude of Gaussian bump
    - params[2]: center frequency of bump (in units of fundamental)
    - params[3]: width (sigma) of bump
    - params[4]: second bump amplitude
    - params[5]: second bump center
    - params[6]: second bump width
    
    The heights are: h[n] = sum_k P(k) * exp(2*pi*i*k*n/N) / N
    We take real part and clip to nonnegative.
    """
    n_params = len(spectral_params)
    
    # Extract parameters
    base = spectral_params[0]
    
    # Build power spectrum P(k) for k = 0, 1, ..., N/2
    n_freq = N // 2 + 1
    k_vals = np.arange(n_freq, dtype=np.float64)
    
    P = np.full(n_freq, base, dtype=np.float64)
    
    # Add Gaussian bumps
    n_bumps = min(3, (n_params - 1) // 3)
    for i in range(n_bumps):
        idx = 1 + 3 * i
        if idx + 2 >= n_params:
            break
        amp = spectral_params[idx]
        center = spectral_params[idx + 1]
        sigma = max(spectral_params[idx + 2], 0.5)
        
        P += amp * np.exp(-((k_vals - center) ** 2) / (2 * sigma ** 2))
    
    # Ensure P is non-negative (power spectrum must be >= 0)
    P = np.maximum(P, 0.0)
    
    # Convert to heights via inverse DFT
    # h[n] = (1/N) * sum_{k=0}^{N/2} P(k) * exp(2*pi*i*k*n/N) * (1 if k==0 or k==N/2 else 2)
    # For real-valued output, use rfft conventions
    
    # Build full spectrum for ifft
    F_full = np.zeros(N, dtype=np.complex128)
    F_full[:n_freq] = P
    # Mirror for negative frequencies
    F_full[n_freq:] = P[N - np.arange(1, N - n_freq + 1)] if N - n_freq > 0 else P[1:-1][::-1]
    
    # Actually, let's use a cleaner approach:
    # h[n] = (2/N) * sum_{k=1}^{N/2-1} P(k)*cos(2*pi*k*n/N) + (P(0) + P(N/2) if N even) / N
    
    if N % 2 == 0:
        n_half = N // 2
        h = np.zeros(N, dtype=np.float64)
        h += P[0] / N  # DC
        if n_half < n_freq:
            h += P[n_half] / N * (-1) ** np.arange(N)  # Nyquist
        for k in range(1, n_half):
            h += (2.0 / N) * P[k] * np.cos(2 * np.pi * k * np.arange(N) / N)
    else:
        n_half = N // 2
        h = np.zeros(N, dtype=np.float64)
        h += P[0] / N
        for k in range(1, n_half + 1):
            h += (2.0 / N) * P[k] * np.cos(2 * np.pi * k * np.arange(N) / N)
    
    # Clip to nonnegative
    h = np.maximum(h, 0.0)
    
    return h

def _spectrum_objective(spectral_params):
    """Objective function for spectral parameter optimization."""
    heights = _spectrum_to_heights(spectral_params)
    if np.sum(heights) <= 0:
        return 1e10
    return compute_score(heights)

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
    W = np.fft.rfft(softmax_weights)
    G = np.fft.irfft(W * np.conj(F), n=M_FFT)
    
    # G[j] = d(soft_max)/d(f_fine[j])
    # Now map to heights
    G_step = G[:M_STEP]
    G_reshaped = G_step.reshape(N, K)
    grad_conv = np.sum(G_reshaped, axis=1)
    
    integral_f = np.sum(heights) * w
    score = soft_max / (integral_f ** 2)
    
    # Gradient of score w.r.t. heights
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
    
    # Strategy 3: Spectral shaping - optimize power spectrum parameters
    if time.time() - START_TIME < TIME_LIMIT - 40:
        # Try different spectral parameterizations
        rng_spec = np.random.default_rng(99)
        
        for n_bumps in [1, 2, 3]:
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
            
            n_params = 1 + 3 * n_bumps  # base + n_bumps * (amp, center, sigma)
            
            for restart in range(5):
                if time.time() - START_TIME > TIME_LIMIT - 15:
                    break
                
                # Random initial spectral parameters
                params0 = np.zeros(n_params)
                params0[0] = rng_spec.uniform(0.5, 2.0)  # base
                for i in range(n_bumps):
                    idx = 1 + 3 * i
                    params0[idx] = rng_spec.uniform(-0.5, 0.5)  # amplitude
                    params0[idx + 1] = rng_spec.uniform(1, N // 4)  # center freq
                    params0[idx + 2] = rng_spec.uniform(2, 20)  # sigma
                
                try:
                    result = minimize(
                        _spectrum_objective,
                        params0,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1e-6}
                    )
                    
                    if result.fun < best_score:
                        h_spec = _spectrum_to_heights(result.x)
                        actual_score = compute_score(h_spec)
                        if actual_score < best_score:
                            best_score = actual_score
                            best_h = h_spec.copy()
                except Exception:
                    pass
    
    # Strategy 4: Projected gradient descent with soft-max smoothing
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
    
    # Strategy 5: Final local search
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
