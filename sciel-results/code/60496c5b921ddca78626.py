import time
import numpy as np
from scipy.optimize import minimize

N = 600
TOTAL_TIME = 150.0
SAFETY_MARGIN = 5.0

def _autoconv_peak(f):
    """Compute the normalized maximum of the autoconvolution of f."""
    dx = 1.0 / (2.0 * N)
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    peak = np.max(conv) * dx
    l1 = np.sum(f) * dx
    normalized_peak = peak / (l1 * l1)
    
    return normalized_peak

def _objective(params):
    """Objective function for optimization: normalized peak of autoconvolution."""
    f = np.exp(params)
    return _autoconv_peak(f)

def _objective_grad(params):
    """Objective function with gradient."""
    f = np.exp(params)
    
    dx = 1.0 / (2.0 * N)
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    peak = np.max(conv) * dx
    l1 = np.sum(f) * dx
    obj = peak / (l1 * l1)
    
    # Compute gradient
    # d(conv)/d(f_i) = conv with f shifted
    # Use FFT to compute gradient efficiently
    # grad_conv[i] = sum_j f[j] * delta(i-j) + sum_j f[j] * delta(i+j)
    # Actually, for autoconvolution: (f*f)[k] = sum_j f[j] f[k-j]
    # d/d(f_i) (f*f)[k] = f[k-i] + f[i-k] (for appropriate ranges)
    
    # Gradient of peak: we need to find where the max occurs
    peak_idx = np.argmax(conv)
    
    # Gradient of conv with respect to f
    # conv[k] = sum_j f[j] * f[k-j] * dx
    # d(conv[k])/d(f[i]) = f[k-i] * dx + f[i-k] * dx (if indices valid)
    
    grad_conv = np.zeros(N)
    for j in range(N):
        # Contribution to conv[peak_idx] from f[j]
        # conv[peak_idx] = sum_j f[j] * f[peak_idx - j]
        # d/d(f[i]) = f[peak_idx - i] + f[i - peak_idx] (by symmetry of autoconv)
        idx1 = peak_idx - j
        idx2 = j - peak_idx
        
        term = 0.0
        if 0 <= idx1 < N:
            term += f[idx1]
        if 0 <= idx2 < N:
            term += f[idx2]
        
        grad_conv[j] = term * dx
    
    # Gradient of objective
    # obj = peak / l1^2
    # d(obj)/d(f[i]) = (d(peak)/d(f[i]) * l1^2 - peak * 2*l1 * d(l1)/d(f[i])) / l1^4
    #                = (d(peak)/d(f[i]) * l1 - 2*peak * d(l1)/d(f[i])) / l1^3
    
    dl1 = dx  # d(l1)/d(f[i]) = dx for all i
    
    grad_obj = (grad_conv * l1 - 2 * peak * dl1) / (l1 ** 3)
    
    # Chain rule: d(params)/d(f[i]) = 1/f[i]
    grad_params = grad_obj / f
    
    return obj, grad_params

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME - SAFETY_MARGIN
    
    # Start with a good initialization
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Try several initializations and pick the best
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Cosine bump
    initializations.append(0.5 + 0.5 * np.cos(np.pi * t))
    
    # Two-bump
    init2 = np.ones(N)
    init2[:N//4] *= 1.5
    init2[3*N//4:] *= 0.5
    initializations.append(init2)
    
    # Gaussian
    initializations.append(np.exp(-20 * t**2))
    
    # Asymmetric
    init4 = 1.0 + 0.3 * np.sin(2 * np.pi * t)
    initializations.append(np.maximum(0, init4))
    
    # Evaluate all and pick best
    best_f = None
    best_val = float('inf')
    
    for init in initializations:
        init = np.maximum(0, init)
        val = _autoconv_peak(init)
        if val < best_val:
            best_val = val
            best_f = init.copy()
    
    # Convert to log-space for unconstrained optimization
    # f = exp(params), so params = log(f)
    # Handle zeros by clipping
    f_safe = np.maximum(best_f, 1e-10)
    params = np.log(f_safe)
    
    # Use L-BFGS-B with gradient
    result = None
    try:
        result = minimize(
            _objective_grad,
            params,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
    except Exception:
        pass
    
    if result is not None and result.success and time.time() < deadline:
        f_opt = np.exp(result.x)
        val_opt = _autoconv_peak(f_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_f = f_opt.copy()
    
    # Final polish with coordinate descent
    if time.time() < deadline - 10:
        f = best_f.copy()
        current_val = best_val
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for idx in range(N):
                if time.time() > deadline - 10:
                    break
                old_val = f[idx]
                for delta in [-0.005, 0.005]:
                    new_val = max(0.0, old_val + delta)
                    f[idx] = new_val
                    new_score = _autoconv_peak(f)
                    if new_score < current_val:
                        current_val = new_score
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_f = f.copy()
                    else:
                        f[idx] = old_val
    
    return list(best_f)

if __name__ == "__main__":
    result = construct()
    print(f"Result: {len(result)} values, sum={sum(result):.4f}")
