import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 1.0 / (2 * N)

def compute_raw_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of nonnegative step heights.
    Score = max(autoconv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-12):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-12:
        return np.inf
    
    # Compute autoconvolution using FFT
    # We need the full convolution length 2N-1
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    # Power spectrum is |F|^2? No, for autoconvolution of real signal f,
    # fft(f*f) = fft(f) * fft(f). Since f is real, fft(f) is conjugate symmetric.
    # The inverse FFT of F*F gives the circular convolution.
    # With n_fft >= 2N-1, it is the linear convolution.
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    
    # The raw convolution values sum to (sum(f))^2.
    # The "normalized maximum" is max(conv) / (sum(f))^2
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """
    Smooth approximation of max(x) using log-sum-exp.
    """
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    # Shift for numerical stability
    x_shifted = x - x_max
    # Avoid overflow in exp
    exp_x = np.exp(beta * x_shifted)
    log_sum = np.log(np.sum(exp_x))
    return x_max + log_sum / beta

def objective_and_grad(f, beta=10.0):
    """
    Returns the objective value (smooth max of normalized conv) and its gradient.
    We optimize with fixed sum(f) = N for stability, then normalize at the end.
    Actually, the objective is scale invariant.
    Let's work with sum(f) = N.
    Obj = max(conv) / N^2.
    Since N is constant, we minimize max(conv).
    Using smooth max: L = smooth_max(conv, beta) / N^2.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-12:
        return 1e10, np.zeros_like(f)
    
    # Normalize to sum N for the gradient step calculation
    f_norm = f * (N / s)
    
    n_fft = 2 * N
    F = np.fft.rfft(f_norm, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    
    # Objective value (raw, will be scaled in final score)
    # We want to minimize max(conv).
    # Smooth max
    max_c = np.max(conv)
    if max_c < 1e-15:
        return 0.0, np.zeros_like(f)
        
    # Weights for smooth max gradient
    # w_k = exp(beta * (conv_k - max_c)) / sum(exp(beta * (conv_j - max_c)))
    beta_safe = min(beta, 100.0) # Cap beta to avoid numerical issues
    shifted = conv - max_c
    exp_shifted = np.exp(beta_safe * shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp < 1e-15:
        return 0.0, np.zeros_like(f)
    weights = exp_shifted / sum_exp
    
    # Gradient of smooth_max(conv) w.r.t f_norm
    # d(smooth_max)/d(conv_k) = weights[k]
    # conv = ifft(F^2)
    # d(conv)/d(F) = 2 * conj(F) * weights_hat ?
    # Let's use the chain rule via FFT.
    # conv = ifft(F * F)
    # dL/df = ifft( dL/dF * dF/df )? No.
    # L = sum_k weights_k conv_k
    # conv = ifft(F^2)
    # dL = sum_k weights_k d(conv_k)
    # In frequency domain:
    # dL = ifft( F^2 ) * weights -> This is circular convolution of weights with conv? No.
    # Let's use: dL/df = 2 * ifft(F * fft(weights))
    # Check:
    # conv_k = sum_i f_i f_{k-i}
    # dL/df_i = sum_k weights_k (f_{k-i} + f_{i-k})
    # This is 2 * (weights * f)_i (cross correlation).
    # Cross correlation can be computed via FFT:
    # fft(weights * f) = fft(weights) * fft(f)? No, that's convolution.
    # Correlation is ifft(fft(weights) * conj(fft(f))).
    # Since weights and f are real, and we are dealing with auto-convolution symmetry,
    # let's compute:
    W = np.fft.rfft(weights, n=n_fft)
    F_full = np.fft.rfft(f_norm, n=n_fft)
    # Gradient is 2 * ifft(W * conj(F_full)) ?
    # Let's verify:
    # (weights * f)_i = sum_k weights_k f_{k-i}
    # ifft(fft(weights) * conj(fft(f)))_i = sum_k weights_k f_{k-i}
    # Yes.
    grad_conv = 2.0 * np.fft.irfft(W * np.conj(F_full), n=n_fft)[:N]
    
    # Scale gradient back to original f scale
    # f_norm = f * (N/s)
    # dL/df = dL/df_norm * (N/s)
    grad_f = grad_conv * (N / s)
    
    # Objective value to return (scaled to match compute_raw_score)
    val = max_c / (s * s)
    # Wait, the gradient computed is for f_norm.
    # If we minimize L(f_norm), and f_norm = f * c, then dL/df = c * dL/df_norm.
    # We returned grad_f scaled by N/s.
    # The value val is max(conv_norm) / N^2 ?
    # conv_norm is based on f_norm (sum N).
    # max(conv_norm) / N^2 is the normalized max for f_norm.
    # Since f and f_norm are proportional, the normalized max is the same.
    # So val is the correct objective value.
    
    return val, grad_f

def spectral_gradient_descent(f_init, start_time, max_time, beta_start=1.0, beta_end=50.0):
    """
    Gradient descent in the frequency/time domain with non-negativity projection.
    """
    f = f_init.copy()
    # Normalize to sum N
    s = np.sum(f)
    if s < 1e-12:
        return f, compute_raw_score(f)
    f = f * (N / s)
    
    best_f = f.copy()
    best_val = compute_raw_score(f)
    
    # Parameters
    lr = 0.01
    min_lr = 1e-8
    max_lr = 0.5
    patience = 0
    max_patience = 50
    
    n_steps = 0
    while time.time() - start_time < max_time:
        # Animate beta: start low (smooth), end high (sharp)
        # Simple linear interpolation based on time
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / max_time)
        beta = beta_start + (beta_end - beta_start) * progress
        
        val, grad = objective_and_grad(f, beta=beta)
        
        if val < best_val:
            best_val = val
            best_f = f.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                # Reduce learning rate
                lr *= 0.5
                if lr < min_lr:
                    break
                patience = 0
        
        # Gradient step
        # We need to project onto the simplex {f >= 0, sum(f) = N}
        # 1. Gradient descent step
        f_new = f - lr * grad
        
        # 2. Project onto f >= 0
        f_new = np.maximum(f_new, 0.0)
        
        # 3. Project onto sum(f) = N
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            # If all zero, reset to a small uniform or break
            break
        f_new = f_new * (N / s_new)
        
        # Check if the step improved the objective (using the same beta for consistency in step acceptance)
        # Note: objective_and_grad uses the current beta.
        val_new, _ = objective_and_grad(f_new, beta=beta)
        
        if val_new < val:
            f = f_new
            # Increase LR slightly if improving
            lr = min(lr * 1.05, max_lr)
        else:
            # Decrease LR
            lr = max(lr * 0.8, min_lr)
            # If stuck, try a smaller step manually?
            # For now, just reduce LR.
            
        n_steps += 1
        if n_steps > 10000: # Safety valve
             break

    return best_f, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess: Gaussian
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # A Gaussian centered at 0
    sigma = 0.1
    f0 = np.exp(-0.5 * (x / sigma)**2)
    
    best_f = None
    best_val = np.inf
    
    # Try a few different initializations
    initializations = [
        np.exp(-0.5 * (x / 0.1)**2),
        np.exp(-0.5 * (x / 0.05)**2),
        np.exp(-0.5 * (x / 0.2)**2),
        np.ones(N) * 0.5 + 0.5 * np.cos(2 * np.pi * x / 0.5), # Cosine bump
    ]
    
    for f_init in initializations:
        if time.time() > deadline - 10:
            break
            
        f_res, val_res = spectral_gradient_descent(f_init, start_time, deadline - start_time - 5)
        
        # Final polish: ensure non-negative and compute exact score
        f_res = np.maximum(f_res, 0.0)
        s = np.sum(f_res)
        if s < 1e-12:
            continue
        f_res = f_res / s # Normalize to sum 1 for final check? 
        # The score is invariant to scaling.
        val_final = compute_raw_score(f_res)
        
        if val_final < best_val:
            best_val = val_final
            best_f = f_res.copy()
    
    if best_f is None:
        # Fallback
        best_f = np.ones(N)
        
    # Return the step heights
    # The task asks for "sequence of nonnegative step heights"
    # The evaluator likely computes the score based on these heights.
    # We should return the heights that minimize the score.
    # The normalization in compute_raw_score handles the scaling.
    # Returning the raw heights is fine.
    return list(best_f)

if __name__ == "__main__":
    f = construct()
    print(f"Length: {len(f)}")
    print(f"Score: {compute_raw_score(np.array(f))}")
