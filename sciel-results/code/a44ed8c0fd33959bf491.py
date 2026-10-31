import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

N = 600
TOTAL_TIME = 150.0
SAFETY_MARGIN = 5.0
dx = 1.0 / (2.0 * N)

def _autoconv_peak(f):
    """Compute the normalized maximum of the autoconvolution of f."""
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    peak = np.max(conv) * dx
    l1 = np.sum(f) * dx
    normalized_peak = peak / (l1 * l1)
    
    return normalized_peak

def _compute_conv(f):
    """Compute the autoconvolution of f on the full grid."""
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    return conv, dx

def _objective_log(params):
    """Objective function using log-parameterization to ensure positivity."""
    f = np.exp(params)
    return _autoconv_peak(f)

def _objective_grad_log(params):
    """Objective function with gradient using log-parameterization."""
    f = np.exp(params)
    val = _autoconv_peak(f)
    
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    peak_val = np.max(conv) * dx
    peak_idx = np.argmax(conv)
    l1 = np.sum(f) * dx
    norm_peak = peak_val / (l1 * l1)
    
    # Gradient of norm_peak with respect to f, then chain rule for log params
    # norm_peak = peak / (l1^2)
    # d(norm_peak)/d(f[i]) = (d(peak)/d(f[i]) * l1^2 - peak * 2*l1 * d(l1)/d(f[i])) / l1^4
    # d(l1)/d(f[i]) = dx
    # d(peak)/d(f[i]) = dx * (2 * sum_j f[j] where conv[peak_idx] involves f[i])
    
    # More carefully:
    # conv[k] = sum_j f[j] * f[k-j] * dx (approximately, in discrete)
    # Actually conv is computed via FFT, so let's think of it as:
    # conv[k] = sum_j f[j] * f[(k-j) mod size]
    # But we only use the first N values of f, padded with zeros.
    
    # For the peak at index k=peak_idx:
    # d(conv[k])/d(f[i]) = sum over j such that j and k-j are both in [0,N)
    # This is essentially the cross-correlation of f with a shifted version.
    
    # Simpler: use finite differences or analytic gradient via FFT
    # The gradient of ||F(f)||_inf^2 is complex, but for the peak:
    
    # Let's use the fact that d(conv[k])/d(f[i]) = 2 * f[(k-i)] if 0 <= k-i < N, else 0
    # (with appropriate handling for the self-term)
    
    grad_f = np.zeros(N)
    k = peak_idx
    for i in range(N):
        j = k - i
        if 0 <= j < N:
            if i == j:
                grad_f[i] += 2.0 * f[i] * dx
            else:
                grad_f[i] += 2.0 * f[j] * dx
    
    # Now chain rule: d(norm_peak)/d(f[i]) = [grad_f[i] * l1^2 - peak_val * 2 * l1 * dx] / l1^4
    # = [grad_f[i] / l1^2 - 2 * peak_val * dx / l1^3]
    # Wait, let me redo:
    # norm_peak = peak_val / (l1 * l1)
    # d(norm_peak)/d(f[i]) = (d(peak_val)/d(f[i]) * l1^2 - peak_val * 2 * l1 * d(l1)/d(f[i])) / l1^4
    # = (grad_f[i] * l1^2 - peak_val * 2 * l1 * dx) / l1^4
    # = grad_f[i] / l1^2 - 2 * peak_val * dx / l1^3
    
    d_norm_peak_df = grad_f / (l1 * l1) - 2.0 * peak_val * dx / (l1 * l1 * l1)
    
    # Chain rule for log parameterization: f = exp(params), so d(f[i])/d(params[i]) = f[i]
    grad_params = d_norm_peak_df * f
    
    return norm_peak, grad_params

def _remez_iteration(f, max_iter=50, tol=1e-6):
    """
    Remez-like algorithm for minimizing the max autoconvolution.
    
    The idea: we want to find f >= 0 such that the autoconvolution is as flat as possible,
    ideally constant. This is a Chebyshev approximation problem.
    
    We alternate between:
    1. Finding the current peak location
    2. Adjusting f to reduce the peak while maintaining non-negativity
    """
    f = f.copy()
    best_f = f.copy()
    best_val = _autoconv_peak(f)
    
    for it in range(max_iter):
        conv, dx_local = _compute_conv(f)
        peak_val = np.max(conv) * dx_local
        peak_idx = np.argmax(conv)
        l1 = np.sum(f) * dx_local
        norm_peak = peak_val / (l1 * l1)
        
        if norm_peak < best_val:
            best_val = norm_peak
            best_f = f.copy()
        
        # Compute gradient of the peak with respect to f
        # Using the analytic gradient from above
        grad_f = np.zeros(N)
        k = peak_idx
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                if i == j:
                    grad_f[i] += 2.0 * f[i]
                else:
                    grad_f[i] += 2.0 * f[j]
        
        # Normalize and smooth
        grad_norm = np.linalg.norm(grad_f)
        if grad_norm > 1e-15:
            grad_f = grad_f / grad_norm
        
        # Smooth the gradient
        kernel = np.ones(10) / 10.0
        smooth_grad = np.convolve(grad_f, kernel, mode='same')
        
        # Update: decrease f where gradient is high, increase where low
        # But ensure non-negativity
        step_size = 0.05
        f_new = f - step_size * (smooth_grad - np.mean(smooth_grad))
        f_new = np.maximum(f_new, 0.0)
        
        # Preserve L1 norm
        l1_old = np.sum(f)
        l1_new = np.sum(f_new)
        if l1_new > 1e-15:
            f_new *= l1_old / l1_new
        
        f = f_new
    
    return best_f, best_val

def _optimize_with_scipy(f_init, deadline):
    """Use scipy.optimize.minimize with L-BFGS-B on log-parameterized variables."""
    params_init = np.log(f_init + 1e-10)
    
    def obj_and_grad(params):
        val, grad = _objective_grad_log(params)
        return val, grad
    
    best_val = _autoconv_peak(f_init)
    best_params = params_init.copy()
    
    try:
        result = minimize(
            obj_and_grad,
            params_init,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass
    
    f_opt = np.exp(best_params)
    f_opt = np.maximum(f_opt, 0.0)
    
    # Normalize to preserve L1
    l1 = np.sum(f_init)
    l1_opt = np.sum(f_opt)
    if l1_opt > 1e-15:
        f_opt *= l1 / l1_opt
    
    return f_opt, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME - SAFETY_MARGIN
    
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Cosine bump
    initializations.append(0.5 + 0.5 * np.cos(np.pi * t))
    
    # Gaussian
    initializations.append(np.exp(-20 * t**2))
    
    # Two-bump
    init2 = np.ones(N)
    init2[:N//4] *= 1.5
    init2[3*N//4:] *= 0.5
    initializations.append(init2)
    
    # Triangular
    init3 = np.abs(1 - 2 * np.abs(t) / 0.5)
    initializations.append(init3)
    
    # Sinc-like
    init4 = np.sinc(5 * t)
    init4 = np.maximum(init4, 0)
    initializations.append(init4)
    
    # Evaluate all and pick best
    best_f = None
    best_val = float('inf')
    
    for init in initializations:
        init = np.maximum(0, init)
        val = _autoconv_peak(init)
        if val < best_val:
            best_val = val
            best_f = init.copy()
    
    # Run scipy optimization on the best initialization
    if time.time() < deadline - 30:
        f_opt, val_opt = _optimize_with_scipy(best_f, deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_f = f_opt.copy()
    
    # Run Remez-like iterations
    if time.time() < deadline - 10:
        f_remez, val_remez = _remez_iteration(best_f, max_iter=100)
        if val_remez < best_val:
            best_val = val_remez
            best_f = f_remez.copy()
    
    # Final polish with coordinate descent
    if time.time() < deadline - 5:
        f = best_f.copy()
        current_val = best_val
        
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5:
                    break
                old_val = f[idx]
                for delta in [-0.002, 0.002]:
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
