import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct
import time

N = 400
MAX_STEPS = 30

def _evaluate(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv)) / (s * s)

def _recurrence_to_h(coeffs, baseline):
    """
    Generate step heights from a linear recurrence.
    coeffs: sparse companion matrix coefficients (length k)
    baseline: constant offset
    
    The recurrence: x[n] = baseline + sum_{j=1}^{k} coeffs[j-1] * x[n-j]
    We initialize with x[0] = 1, x[n<0] = 0, and iterate forward.
    This ensures exponentially decaying memory if |eigenvalues| < 1.
    """
    k = len(coeffs)
    h = np.zeros(N)
    h[0] = 1.0
    for n in range(1, N):
        val = baseline
        for j in range(1, k + 1):
            if n - j >= 0:
                val += coeffs[j-1] * h[n-j]
        h[n] = val
    return h

def _recurrence_obj(params, beta=50.0):
    """
    Objective for optimizing recurrence coefficients and baseline.
    params: [coeffs..., baseline]
    """
    k = MAX_STEPS
    coeffs = params[:k]
    baseline = params[k]
    
    # Check stability: eigenvalues of companion matrix should be inside unit circle
    # Companion matrix for x[n] = c1*x[n-1] + ... + ck*x[n-k]
    # The characteristic polynomial is z^k - c1*z^(k-1) - ... - ck
    # We check if the recurrence is stable by examining the coefficients
    # A simple check: if sum of abs coeffs > 1, likely unstable
    if np.sum(np.abs(coeffs)) > 1.5:
        return 1e10
    
    h = _recurrence_to_h(coeffs, baseline)
    
    # Check finiteness
    if not np.all(np.isfinite(h)):
        return 1e10
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    conv = np.convolve(h, h, mode='full')
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _recurrence_grad(params, beta=50.0):
    """
    Gradient for recurrence optimization.
    params: [coeffs..., baseline]
    """
    k = MAX_STEPS
    coeffs = params[:k]
    baseline = params[k]
    
    if np.sum(np.abs(coeffs)) > 1.5:
        return np.zeros_like(params)
    
    h = _recurrence_to_h(coeffs, baseline)
    
    if not np.all(np.isfinite(h)):
        return np.zeros_like(params)
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(params)
    
    conv = np.convolve(h, h, mode='full')
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(params)
    
    # Gradient of soft_max w.r.t. h
    grad_h = _grad_autoconv(weights, h)
    
    # Gradient of objective w.r.t. h
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_h_i * s - 2*soft_max) / s^3
    grad_obj_h = 2.0 * N * (grad_h * s - 2.0 * soft_max) / (s * s * s)
    
    # Now propagate gradient back through the recurrence
    # h[n] = baseline + sum_j coeffs[j-1] * h[n-j]
    # dh[n]/d(coeffs[j-1]) = h[n-j] for n >= j, else 0
    # dh[n]/d(baseline) = 1 for all n
    
    grad_coeffs = np.zeros(k)
    grad_baseline = 0.0
    
    # For each h[n], we need to accumulate gradients
    # dh[n]/d(coeffs[j-1]) = h[n-j] if n >= j
    for n in range(N):
        for j in range(1, k + 1):
            if n >= j:
                grad_coeffs[j-1] += grad_obj_h[n] * h[n-j]
        grad_baseline += grad_obj_h[n]
    
    return np.concatenate([grad_coeffs, [grad_baseline]])

def _soft_max_and_weights(conv, beta):
    """Compute soft-max and its weights for the autoconvolution."""
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    return soft_max, weights

def _grad_autoconv(w, h):
    """
    Gradient of sum_j w_j * conv_j w.r.t. h.
    conv = full autoconvolution of h, conv[j] = sum_l h[l]*h[j-l] for j in 0..2N-2.
    d(conv[j])/d(h[i]) = 2*h[j-i] if 0 <= j-i < N, else 0.
    So grad_h[i] = 2 * sum_{j: 0<=j-i<N} w[j]*h[j-i]
                 = 2 * sum_{k=0}^{N-1} w[i+k]*h[k]  (where k=j-i, so j=i+k)
    This is 2 * (cross-correlation of w and h)[i] where cross_corr[i] = sum_k w[i+k]*h[k].
    Using FFT with zero-padding to avoid circular wrap-around.
    """
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(w)] = w
    h_pad = np.zeros(L)
    h_pad[:N] = h
    
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    
    # Cross-correlation: sum_k w[i+k]*h[k]
    # This is ifft(W * conj(H))
    cross = np.fft.ifft(W * np.conj(H)).real
    grad_h = 2.0 * cross[:N]
    return grad_h

def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(h)
    
    grad_sm = _grad_autoconv(weights, h)
    
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_sm_i * s^2 - soft_max * 2*s) / s^4
    #         = 2*N * (grad_sm_i * s - 2*soft_max) / s^3
    return 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)

def _epigraph_obj(params, beta=50.0):
    """Epigraph formulation: optimize over (h, t) where t >= soft_max(conv(h))/s^2"""
    h = params[:N]
    t = params[N]
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    soft_max, _ = _soft_max_and_weights(conv, beta)
    target = soft_max / (s * s)
    # Penalty for t < target
    if t < target:
        return 2.0 * N * target + 100.0 * (target - t) ** 2
    return 2.0 * N * t

def _epigraph_grad(params, beta=50.0):
    h = params[:N]
    t = params[N]
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(params)
    
    conv = np.convolve(h, h, mode='full')
    soft_max, weights = _soft_max_and_weights(conv, beta)
    target = soft_max / (s * s)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(params)
    
    grad_sm = _grad_autoconv(weights, h)
    grad_target = (grad_sm * s - 2.0 * soft_max) / (s * s * s)
    
    if t < target:
        grad_h = 2.0 * N * grad_target + 200.0 * (target - t) * grad_target
        grad_t = -200.0 * (target - t)
    else:
        grad_h = np.zeros_like(h)
        grad_t = 2.0 * N
    
    return np.concatenate([grad_h, [grad_t]])

def _spectral_penalty(h, lam=0.1):
    """Penalty for spectral concentration (L4 norm of DCT coefficients)."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    return lam * np.sum(np.abs(dct_coeffs)**4) / N

def _spectral_penalty_grad(h, lam=0.1):
    """Gradient of spectral penalty."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    grad = 4.0 * lam / N * idct(dct_coeffs**3, type=2, norm='ortho')
    return grad

def _combined_obj(h, beta=50.0, lam=0.05):
    return _objective_soft(h, beta) + _spectral_penalty(h, lam)

def _combined_grad(h, beta=50.0, lam=0.05):
    return _objective_soft_grad(h, beta) + _spectral_penalty_grad(h, lam)

def _refine(h, deadline):
    """Refine a candidate h using a sequence of optimization phases."""
    h_curr = h.copy()
    
    # Phase 1: Moderate beta, spectral flattening
    if time.time() < deadline - 15.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(30.0, 0.1), jac=_combined_grad,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 2: Higher beta, less spectral penalty (focus on peak)
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(80.0, 0.01), jac=_combined_grad,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 3: Epigraph formulation for better scale optimization
    if time.time() < deadline - 5.0:
        try:
            s = np.sum(h_curr)
            t_init = np.max(np.convolve(h_curr, h_curr, mode='full')) / (s * s) if abs(s) > 1e-10 else 1.0
            params_init = np.concatenate([h_curr, [t_init]])
            res = minimize(_epigraph_obj, params_init, method='L-BFGS-B',
                           args=(100.0,), jac=_epigraph_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x[:N]
        except Exception:
            pass
    
    # Phase 4: Pure peak polishing with very high beta
    if time.time() < deadline - 2.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(200.0,), jac=_objective_soft_grad,
                           options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11})
            h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Strategy 1: Recurrence-based optimization
    # Initialize with small coefficients for stability
    for trial in range(10):
        if time.time() > deadline - 30.0:
            break
        
        # Initialize recurrence coefficients
        if trial == 0:
            # Start with simple decay: x[n] = 0.9 * x[n-1]
            coeffs_init = np.zeros(MAX_STEPS)
            coeffs_init[0] = 0.9
            baseline_init = 0.1
        elif trial == 1:
            # Two-term recurrence
            coeffs_init = np.zeros(MAX_STEPS)
            coeffs_init[0] = 0.5
            coeffs_init[1] = 0.4
            baseline_init = 0.2
        elif trial == 2:
            # Alternating pattern
            coeffs_init = np.zeros(MAX_STEPS)
            coeffs_init[0] = -0.3
            coeffs_init[1] = 0.5
            baseline_init = 0.3
        else:
            # Random small coefficients
            coeffs_init = rng.standard_normal(MAX_STEPS) * 0.1
            baseline_init = rng.uniform(0.1, 0.5)
        
        params_init = np.concatenate([coeffs_init, [baseline_init]])
        
        try:
            res = minimize(_recurrence_obj, params_init, method='L-BFGS-B',
                           args=(50.0,), jac=_recurrence_grad,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
            h_rec = _recurrence_to_h(res.x[:MAX_STEPS], res.x[MAX_STEPS])
            if np.all(np.isfinite(h_rec)) and abs(np.sum(h_rec)) > 1e-10:
                val_rec = _evaluate(h_rec)
                if val_rec < best_val:
                    best_val = val_rec
                    best_h = h_rec.copy()
        except Exception:
            pass
    
    # Strategy 2: Traditional optimization on best found
    if time.time() < deadline - 10.0:
        h_refined = _refine(best_h, deadline)
        val = _evaluate(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
    
    return best_h
