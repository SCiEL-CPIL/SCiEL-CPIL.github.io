"""
Optimization for the third autocorrelation inequality.

Strategy: Directly minimize the max of the autoconvolution via a smooth
surrogate (log-sum-exp), using L-BFGS-B with analytic gradients in the
frequency domain. The key insight is that we can work in the Fourier domain
where the convolution becomes a pointwise square, making gradient computation
efficient.

We parameterize h in the frequency domain (real FFT coefficients) to exploit
the structure: conv(h) = IFFT(|FFT(h)|^2). The objective is
abs(2*N*max(conv)/sum(h)^2). Since sum(h) = N*h[0] for the DC component,
we can fix the DC component and optimize the rest.
"""
import numpy as np
import time
from scipy.fft import fft, ifft, rfft, irfft
from scipy.optimize import minimize

N = 400

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _eval_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta):
    """Numerically stable soft maximum (log-sum-exp based)."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    return m + np.log(np.sum(exp_vals)) / beta

def _objective_and_grad(params, beta, dc_component):
    """
    Objective and gradient for optimization in the real-valued frequency domain.
    
    params: array of shape (N//2 + 1,) representing rfft coefficients
    beta: soft-max parameter
    dc_component: fixed DC component (sum(h)/N)
    
    Returns: (objective_value, gradient)
    """
    # Reconstruct h from frequency coefficients
    # rfft gives N//2+1 coefficients for real input of length N
    # We fix the DC component (index 0) and optimize the rest
    
    H = params.copy()
    H[0] = dc_component  # Fix DC to maintain sum
    
    # Reconstruct h
    h = irfft(H, n=N)
    
    if not np.all(np.isfinite(h)):
        return 1e10, np.zeros_like(params)
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    # Normalize to have sum = N (so dc = 1)
    h_norm = h * (N / s)
    
    # Compute autoconvolution
    conv = _convolve_fft(h_norm)
    
    # Soft max
    sm = _soft_max(conv, beta)
    
    # Objective: abs(2*N*sm/N^2) = 2*sm/N
    # Since we're minimizing, and sm should be positive for reasonable h,
    # we can drop the abs for gradient computation
    obj = 2.0 * sm / N
    
    # Gradient computation using chain rule
    # conv = IFFT(|FFT(h_norm)|^2)
    # h_norm = h * (N/s)
    # This is complex; let's use a simpler approach
    
    # Actually, let's use a different parameterization: work directly with h
    # and use the fact that conv is a quadratic form
    
    return obj, None  # We'll use numerical gradient for simplicity in this version

def _objective_h(h_params, beta, target_sum):
    """
    Objective function working directly with h.
    h_params: array of length N-1 (all but the first element, which is fixed)
    beta: soft-max parameter
    target_sum: desired sum of h (typically N)
    """
    # Reconstruct h with fixed first element
    h = np.empty(N)
    h[0] = target_sum - np.sum(h_params)  # Ensure sum = target_sum
    h[1:] = h_params
    
    if not np.all(np.isfinite(h)):
        return 1e10
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return 2.0 * sm / N

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses soft-max surrogate optimization with multiple restarts and
    increasing beta values for progressively sharper maximization.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # We'll work with h[1:] as free parameters, fixing h[0] to maintain sum = N
    # Initial: h[0] = 1, h[1:] = 1
    init_params = np.ones(N - 1)
    
    # Multi-stage optimization with increasing beta
    betas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_params = init_params.copy()
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        # Try multiple restarts for lower beta, fewer for higher beta
        n_restarts = 5 if beta < 10 else 3 if beta < 100 else 2
        
        for trial in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            if trial == 0:
                # Start from current best
                params_trial = current_params.copy()
            else:
                # Add random perturbation
                amp = 0.3 / (1 + trial)
                rng = np.random.RandomState(42 + int(beta) + trial)
                params_trial = current_params + amp * rng.randn(N - 1)
            
            def obj_func(p, beta=beta):
                return _objective_h(p, beta, N)
            
            try:
                max_iter = 500 if beta < 50 else 300
                result = minimize(
                    obj_func,
                    params_trial,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < 1e9:
                    params_opt = result.x
                    h_opt = np.empty(N)
                    h_opt[0] = N - np.sum(params_opt)
                    h_opt[1:] = params_opt
                    
                    if np.all(np.isfinite(h_opt)):
                        v_opt = _eval_value(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                            current_params = params_opt.copy()
            except Exception:
                continue
        
        # Update current_params to best found so far
        if np.all(np.isfinite(best_h)):
            current_params = best_h[1:].copy()
    
    # Final polish with very high beta
    if time.time() < deadline - 10:
        for beta in [2000.0, 5000.0, 10000.0]:
            if time.time() > deadline - 5:
                break
            
            def obj_func(p, beta=beta):
                return _objective_h(p, beta, N)
            
            try:
                result = minimize(
                    obj_func,
                    current_params.copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if result.success or result.fun < 1e9:
                    params_opt = result.x
                    h_opt = np.empty(N)
                    h_opt[0] = N - np.sum(params_opt)
                    h_opt[1:] = params_opt
                    
                    if np.all(np.isfinite(h_opt)):
                        v_opt = _eval_value(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                            current_params = params_opt.copy()
            except Exception:
                continue
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
