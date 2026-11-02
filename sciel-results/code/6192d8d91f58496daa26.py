import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

# Precompute FFT setup
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
    
    score = max_conv / (integral_f ** 2)
    return score

def _smooth_max_objective_and_grad(h, beta=50.0):
    """
    Smooth-max surrogate for the normalized autoconvolution peak.
    Uses log-sum-exp smoothing with parameter beta.
    h: array of N nonnegative step heights (will be clipped internally)
    Returns (objective_value, gradient_wrt_h)
    """
    h = np.maximum(h, 0.0)
    n = len(h)
    w = 0.5 / n
    
    # Build fine grid and compute autoconvolution
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(h, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    integral_f = np.sum(h) * w
    if integral_f <= 0:
        return 1e10, np.zeros_like(h)
    
    norm_conv = conv / (integral_f ** 2)
    
    # Smooth max using log-sum-exp
    # Focus on the central region where the peak typically occurs
    center = M_FFT // 2
    # The autoconvolution of a function supported on [-1/4, 1/4] 
    # has support on [-1/2, 1/2], so in our FFT grid (length M_FFT),
    # the interesting region is around the center
    # Actually, let's just use all of conv but focus on the top values
    
    # For efficiency, use a subset of the convolution values
    # The peak of autoconvolution for a nonnegative function is at lag 0
    # which corresponds to index 0 in the circular convolution
    # But we need to be careful about the FFT convention
    
    # Let's use the full conv array for the smooth max
    # But to save memory/time, we can use a reduced set
    # Actually M_FFT = 32768 is manageable
    
    # Shift so that lag 0 is at index 0
    conv_shifted = np.fft.fftshift(conv)
    
    # The peak should be near the center of conv_shifted
    # Use a window around the center
    half_win = M_FFT // 4
    center_idx = M_FFT // 2
    start_idx = max(0, center_idx - half_win)
    end_idx = min(M_FFT, center_idx + half_win)
    conv_window = conv_shifted[start_idx:end_idx]
    
    # Normalize
    norm_window = conv_window / (integral_f ** 2)
    
    # Smooth max: (1/beta) * log(sum(exp(beta * x_i)))
    beta_safe = max(beta, 1.0)
    max_val = np.max(norm_window)
    exp_terms = np.exp(beta_safe * (norm_window - max_val))
    log_sum = np.log(np.sum(exp_terms))
    smooth_max_val = max_val + log_sum / beta_safe
    
    # Gradient of smooth max w.r.t. conv_window
    # d(smooth_max)/d(conv_window[i]) = exp(beta * (x_i - max)) / sum(exp(beta * (x_j - max)))
    grad_conv_window = exp_terms / np.sum(exp_terms)
    
    # Gradient w.r.t. conv (in the window)
    grad_conv = np.zeros(M_FFT)
    grad_conv[start_idx:end_idx] = grad_conv_window
    
    # Gradient w.r.t. conv_shifted
    grad_conv_shifted = grad_conv
    
    # Gradient w.r.t. conv (undo fftshift)
    grad_conv_unshifted = np.fft.ifftshift(grad_conv_shifted)
    
    # Now we need gradient of conv w.r.t. h
    # conv = IFFT(FFT(f_fine)^2)
    # d(conv)/d(f_fine[j]) involves the FFT
    
    # conv[m] = (1/M_FFT) * sum_k F[k]^2 * exp(2*pi*i*k*m/M_FFT)
    # where F = FFT(f_fine)
    
    # d(conv[m])/d(f_fine[j]) = (1/M_FFT) * sum_k 2*F[k]*dF[k]/df[j] * exp(2*pi*i*k*m/M_FFT)
    # dF[k]/df[j] = exp(-2*pi*i*k*j/M_FFT)
    
    # So d(conv[m])/d(f_fine[j]) = (2/M_FFT) * sum_k F[k] * exp(2*pi*i*k*(m-j)/M_FFT)
    # = (2/M_FFT) * IFFT(F * exp(2*pi*i*k*m/M_FFT)) evaluated at j... 
    # This is getting complex. Let's use a simpler approach.
    
    # Alternative: compute gradient via the chain rule more carefully
    # Let's use the fact that conv = IFFT(|FFT(f)|^2) for real f
    # Actually conv = IFFT(FFT(f) * FFT(f)) for the linear convolution embedded in circular
    
    # For the gradient, we can use:
    # grad_f = FFT^{-1}(2 * F * FFT(grad_conv_unshifted))
    # Wait, that's not quite right either.
    
    # Let me think again. 
    # conv = ifft(F * F) where F = fft(f)
    # d(conv)/df = ifft(2 * F * fft(d(conv)/dF)) ... no
    # 
    # Actually, using the chain rule with FFT:
    # Let g = F * F, then conv = ifft(g)
    # d(conv)/df = fft^{-1}(dg/df) where dg/df is computed element-wise in freq domain
    # dg[k] = 2*F[k]*dF[k]/df
    # dF[k]/df[j] = exp(-2*pi*i*k*j/M_FFT)
    # So dg[k]/df[j] = 2*F[k]*exp(-2*pi*i*k*j/M_FFT)
    # d(conv[m])/df[j] = (1/M_FFT)*sum_k dg[k]/df[j] * exp(2*pi*i*k*m/M_FFT)
    # = (2/M_FFT)*sum_k F[k]*exp(2*pi*i*k*(m-j)/M_FFT)
    
    # This is a convolution of F with itself, which is conv itself!
    # So d(conv[m])/df[j] = (2/M_FFT)*conv[m-j] (with circular indexing)
    
    # Therefore: grad_f[j] = sum_m grad_conv[m] * d(conv[m])/df[j]
    # = (2/M_FFT) * sum_m grad_conv[m] * conv[m-j]
    # = (2/M_FFT) * (grad_conv * conv)[j] (circular cross-correlation)
    
    # So grad_f = (2/M_FFT) * ifft(fft(grad_conv) * fft(conv))
    # Wait, let me be more careful with the FFT conventions.
    
    # Actually, let's just use a simpler numerical approach for the gradient
    # since the analytical derivation is error-prone.
    
    # Use finite differences with a small perturbation
    eps = 1e-6
    grad_h = np.zeros_like(h)
    
    # For efficiency, we'll compute the gradient using the linearity
    # The autoconvolution is quadratic in h, so we can compute the gradient
    # more efficiently.
    
    # Let's use the fact that for a step function:
    # conv at lag l = sum_i h[i] * h[i + l] * w^2 (approximately)
    # But with FFT it's more complex.
    
    # Let's just do a simple finite difference for now, but only for a subset
    # to save time. Actually, let's try the analytical approach.
    
    # The autoconvolution of the step function h (with width w per step) is:
    # (f*f)(t) = integral f(s) f(t-s) ds
    # For a step function, this is piecewise linear.
    
    # Let's use a different approach: compute the gradient via the FFT
    # more carefully.
    
    # f_fine is the fine-grained representation.
    # F = fft(f_fine)
    # conv = ifft(F * F)
    
    # d(conv)/d(f_fine) can be computed as:
    # Let G = fft(grad_conv_unshifted)
    # Then d(conv)/d(f_fine) = ifft(2 * F * G)
    # Wait, I need to verify this.
    
    # conv = ifft(F^2)
    # d(conv) = ifft(2*F*dF)
    # dF = fft(df)
    # So d(conv) = ifft(2*F*fft(df))
    # <d(conv), grad_conv> = <ifft(2*F*fft(df)), grad_conv>
    # = <2*F*fft(df), fft(grad_conv)>  (Parseval)
    # = <df, ifft(2*conj(F)*fft(grad_conv))>  (by adjoint of fft)
    # So grad_f_fine = ifft(2 * conj(F) * fft(grad_conv_unshifted))
    
    G = np.fft.rfft(grad_conv_unshifted)
    grad_f_fine = np.fft.irfft(2 * np.conj(F) * G, n=M_FFT)
    
    # Now aggregate from fine grid to step heights
    # f_fine[i*K:(i+1)*K] = h[i]
    # So d(f_fine[j])/d(h[i]) = 1 if j in [i*K, (i+1)*K), else 0
    # grad_h[i] = sum_{j in block i} grad_f_fine[j]
    
    grad_h = np.zeros(N)
    for i in range(N):
        start = i * K
        end = start + K
        if end <= M_STEP:
            grad_h[i] = np.sum(grad_f_fine[start:end])
    
    # Also need gradient of the normalization factor
    # score = max_conv / integral_f^2
    # d(score)/d(h) = (d(max_conv)/d(h) * integral_f^2 - max_conv * 2 * integral_f * d(integral_f)/d(h)) / integral_f^4
    # = d(max_conv)/d(h) / integral_f^2 - 2 * max_conv * d(integral_f)/d(h) / integral_f^3
    
    d_int = w * np.ones(N)
    
    grad_h = grad_h / (integral_f ** 2) - 2 * smooth_max_val * d_int / (integral_f ** 3)
    
    return smooth_max_val, grad_h

def _lbfgs_optimize(h0, max_iter=200, beta=50.0, time_budget=60.0):
    """Optimize using L-BFGS-B with smooth-max objective."""
    start_t = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_t > time_budget:
            return 1e10, np.zeros_like(x)
        val, grad = _smooth_max_objective_and_grad(x, beta=beta)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(x)
        return val, grad
    
    # Use L-BFGS-B with bounds [0, inf)
    bounds = [(0, None)] * N
    
    try:
        result = minimize(
            obj_and_grad,
            h0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 2,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'disp': False
            }
        )
    except Exception:
        return h0, compute_score(h0)
    
    h_opt = np.maximum(result.x, 0.0)
    score = compute_score(h_opt)
    
    return h_opt, score

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
    
    # Strategy 2: Random smooth perturbations for good initializations
    rng = np.random.default_rng(42)
    initial_candidates = []
    for trial in range(30):
        if time.time() - START_TIME > TIME_LIMIT - 40:
            break
        
        g = np.zeros(N)
        num_modes = rng.integers(2, 8)
        for _ in range(num_modes):
            k = rng.integers(1, N // 6)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.05, 0.25)
            g += amp * np.cos(2 * np.pi * k * x + phase)
        
        g -= np.mean(g)
        
        for eps in [0.1, 0.2, 0.3, 0.4]:
            h_trial = 1.0 + eps * g
            h_trial = np.maximum(h_trial, 0.0)
            if np.sum(h_trial) <= 0:
                continue
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
            initial_candidates.append(h_trial.copy())
    
    # Strategy 3: L-BFGS-B optimization from multiple starting points
    # Use the best few candidates as starting points
    # Sort candidates by score
    candidate_scores = []
    for h_cand in initial_candidates:
        s = compute_score(h_cand)
        candidate_scores.append((s, h_cand))
    candidate_scores.sort(key=lambda x: x[0])
    
    # Take the top few unique starting points
    start_points = [best_h.copy()]
    for s, h_cand in candidate_scores[:5]:
        # Check if it's sufficiently different from existing start points
        is_new = True
        for existing in start_points:
            if np.linalg.norm(h_cand - existing) / max(np.linalg.norm(existing), 1e-10) < 0.1:
                is_new = False
                break
        if is_new:
            start_points.append(h_cand.copy())
    
    # Run L-BFGS-B from each starting point
    remaining_time = TIME_LIMIT - (time.time() - START_TIME)
    num_starts = min(len(start_points), 3)
    time_per_start = max(10, (remaining_time - 10) / max(num_starts, 1))
    
    for i, h_start in enumerate(start_points[:num_starts]):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        time_budget = min(time_per_start, TIME_LIMIT - (time.time() - START_TIME) - 10)
        if time_budget < 5:
            break
        
        h_opt, score = _lbfgs_optimize(h_start, max_iter=100, beta=50.0, time_budget=time_budget)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Strategy 4: Final local search with small perturbations
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng2 = np.random.default_rng(123)
        for _ in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng2.normal(0, 0.01, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
