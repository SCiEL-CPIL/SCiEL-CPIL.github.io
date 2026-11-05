"""Rewritten program for third autocorrelation inequality task."""
N = 400
import numpy as np
import time
from scipy.optimize import minimize

def _soft_max(x, beta):
    """Soft maximum using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return x_max
    shifted = beta * (x - x_max)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return x_max + log_sum / beta

def _objective_and_grad(h, beta):
    """Compute objective value and gradient using FFT."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    # Full autoconvolution via FFT
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
    
    # Soft max surrogate
    sm = _soft_max(conv, beta)
    
    # Objective: |2*N*sm / s^2|, we minimize the positive version
    # Since we want to minimize the absolute value, and sm could be negative,
    # let's minimize (2*N*sm/s^2)^2 to handle both signs, or just minimize sm/s^2
    # Actually, the evaluator uses abs(...), so we should minimize the magnitude.
    # But for smoothness, let's minimize (2*N*sm/s^2)^2
    val = 2.0 * N * sm / (s * s)
    obj = val * val
    
    # Gradient via FFT
    # d(conv[k])/d(h[i]) = h[i+k] + h[i-k] (with bounds)
    # This is complex to compute directly. Use numerical or analytical.
    # For efficiency, use the fact that conv = ifft(fft(h)^2)
    # d(conv)/d(h) can be computed via the convolution structure
    
    # Simpler: use central differences for gradient if FFT gradient is complex
    # But that's slow. Let's use the analytical gradient.
    
    # The gradient of soft_max(conv, beta) w.r.t. conv is:
    # exp(beta*(conv[k] - conv_max)) / sum_j exp(beta*(conv[j] - conv_max))
    x_max = np.max(conv)
    shifted = beta * (conv - x_max)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # d(sm)/d(conv[k]) = weights[k]
    # d(conv[k])/d(h[i]) = h[i+k] + h[i-k] if in bounds, else just one term
    # conv[k] = sum_{i} h[i] * h[i+k] for k>=0, with valid ranges
    # Actually conv[k] = sum_i h[i]*h[i+k] where 0<=i<N and 0<=i+k<N
    # So d(conv[k])/d(h[i]) = h[i+k] (if i+k<N) + h[i-k] (if i-k>=0)
    
    # Build gradient: grad_h[i] = sum_k weights[k] * d(conv[k])/d(h[i])
    # = sum_k weights[k] * (h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
    
    # This is a convolution-like operation. Let's compute it efficiently.
    # For each i, grad_h[i] = sum_{k} weights[k] * h[i+k] (k from 0 to N-1-i) 
    #                   + sum_{k} weights[k] * h[i-k] (k from 1 to i)
    
    # This is essentially: grad_h = convolve(h, weights) + convolve(h, weights[::-1])
    # But weights is defined on k=0..2N-2, and the convolution needs care.
    
    # Let's just do it directly but vectorized where possible.
    # For N=400, O(N^2) = 160k operations, which is fast enough.
    
    grad_h = np.zeros(N)
    for i in range(N):
        # Contribution from h[i+k] terms: k such that 0 <= i+k < N => k in [0, N-1-i]
        if N - 1 - i >= 0:
            k_range = np.arange(0, N - i)
            grad_h[i] += np.sum(weights[k_range] * h[i + k_range])
        # Contribution from h[i-k] terms: k such that 0 <= i-k < N => k in [1, i]
        if i >= 1:
            k_range = np.arange(1, i + 1)
            grad_h[i] += np.sum(weights[k_range] * h[i - k_range])
    
    # Now chain rule: d(obj)/d(h[i]) = 2*val * d(val)/d(h[i])
    # val = 2*N*sm/s^2
    # d(val)/d(h[i]) = 2*N * (d(sm)/d(h[i]) * s^2 - sm * 2*s * d(s)/d(h[i])) / s^4
    # = 2*N * (d(sm)/d(h[i]) / s - 2*sm * (1/s) * (1/s^2)) ... let me redo
    # val = 2*N*sm*s^{-2}
    # d(val)/d(h[i]) = 2*N*(d(sm)/d(h[i])*s^{-2} + sm*(-2)*s^{-3}*1)
    # = 2*N*s^{-3}*(d(sm)/d(h[i])*s - 2*sm)
    
    dsm_dh = grad_h  # This is d(sm)/d(h[i]) since d(sm)/d(conv)=weights and we computed d(conv)/d(h)
    
    dval_dh = 2.0 * N * (dsm_dh * s - 2.0 * sm) / (s ** 3)
    grad_obj = 2.0 * val * dval_dh
    
    return obj, grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = _objective_and_grad(best_h, 60.0)[0]
    # Actually compute the true value for seed
    conv_seed = np.convolve(best_h, best_h)
    true_val = abs(2 * N * np.max(conv_seed) / np.sum(best_h)**2)
    best_true_val = true_val
    best_h = best_h.copy()
    
    # Generate diverse initializations
    inits = []
    
    # 1. All ones (baseline)
    inits.append(np.ones(N))
    
    # 2. Random Gaussian
    rng = np.random.RandomState(42)
    inits.append(rng.randn(N))
    
    # 3. Block pattern: alternating blocks
    block_size = 10
    n_blocks = N // block_size
    for trial in range(3):
        signs = rng.choice([-1, 1], size=n_blocks)
        h = np.repeat(signs, block_size)
        inits.append(h + 0.01 * rng.randn(N))
    
    # 4. Gaussian-filtered random
    for trial in range(3):
        h = rng.randn(N)
        # Simple Gaussian smoothing
        kernel = np.exp(-0.5 * (np.arange(-5, 6) / 2.0)**2)
        kernel /= kernel.sum()
        h = np.convolve(h, kernel, mode='same')
        inits.append(h)
    
    # 5. Cosine-based patterns
    for freq in [1, 2, 3, 5]:
        x = np.linspace(0, 2*np.pi*freq, N)
        inits.append(np.cos(x) + 0.01*rng.randn(N))
    
    # 6. Sawtooth-like
    x = np.linspace(0, 1, N)
    inits.append(2*x - 1 + 0.01*rng.randn(N))
    
    # 7. Sparse spikes
    h = np.zeros(N)
    for _ in range(5):
        idx = rng.randint(0, N)
        h[idx] = rng.choice([-1, 1])
    inits.append(h + 0.01*rng.randn(N))
    
    # 8. Low-frequency sine
    x = np.linspace(0, 2*np.pi, N)
    inits.append(np.sin(x) + 0.01*rng.randn(N))
    
    # 9. Two-block: positive then negative
    inits.append(np.concatenate([np.ones(N//2), -np.ones(N - N//2)]))
    
    # 10. Three-block
    inits.append(np.concatenate([np.ones(N//3), np.zeros(N//3), -np.ones(N - 2*(N//3))]))
    
    # Filter out inits with zero sum
    valid_inits = []
    for h in inits:
        if abs(np.sum(h)) > 1e-6:
            valid_inits.append(h)
    
    beta_values = [40, 60, 80, 100]
    
    for init in valid_inits:
        if time.time() > deadline - 5:
            break
        
        for beta in beta_values:
            if time.time() > deadline - 5:
                break
            
            try:
                # First evaluate the initial point
                obj, grad = _objective_and_grad(init, beta)
                
                # Use L-BFGS-B
                res = minimize(
                    _objective_and_grad,
                    init.copy(),
                    jac=True,
                    method='L-BFGS-B',
                    args=(beta,),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.success or res.fun < obj:
                    h_opt = res.x
                    # Compute true value
                    conv_opt = np.convolve(h_opt, h_opt)
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-12:
                        true_v = abs(2 * N * np.max(conv_opt) / (s_opt * s_opt))
                        if true_v < best_true_val:
                            best_true_val = true_v
                            best_h = h_opt.copy()
            except Exception:
                continue
    
    # Final polish: try a few more optimizations from the best point with different betas
    if time.time() < deadline - 3:
        for beta in [80, 100, 120]:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(
                    _objective_and_grad,
                    best_h.copy(),
                    jac=True,
                    method='L-BFGS-B',
                    args=(beta,),
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                if res.fun < _objective_and_grad(best_h, beta)[0]:
                    h_opt = res.x
                    conv_opt = np.convolve(h_opt, h_opt)
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-12:
                        true_v = abs(2 * N * np.max(conv_opt) / (s_opt * s_opt))
                        if true_v < best_true_val:
                            best_true_val = true_v
                            best_h = h_opt.copy()
            except Exception:
                continue
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
