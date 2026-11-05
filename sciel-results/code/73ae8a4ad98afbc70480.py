import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def _autoconv_fft(h):
    """Full autoconvolution via FFT (faster than np.convolve for large N)."""
    n = len(h)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    conv = np.fft.irfft(H * H, fsize)[:size]
    return conv

def _soft_max(x, beta=100.0):
    """Smooth approximation to max(x)."""
    m = np.max(x)
    if not np.isfinite(m):
        return 1e10
    e = np.exp(-beta * (x - m))
    s = np.sum(e)
    if s < 1e-300:
        return m
    return m + np.log(s) / beta

def _objective_and_grad(h, beta=100.0):
    """
    Objective: 2*N * max(conv) / sum(h)^2, where conv = h*h (full autoconv)
    We minimize this. Use soft-max for smoothness.
    
    Let S = sum(h), M = soft_max(conv)
    f = 2*N*M / S^2
    
    df/dh_i = 2*N * [ (dM/dh_i * S^2 - M * 2*S * dh_i_sum) / S^4 ]
            = 2*N * [ dM/dh_i / S^2 - 2*M * (1/S) * (1/S^2) * S ]
            = 2*N * [ dM/dh_i / S^2 - 2*M / S^3 ]
    
    dM/dh_i: for soft_max, dM/dx_j = exp(-beta*(x_j - m)) / sum(exp(...))
    d(conv[j])/d(h[i]) = h[j-i] + h[i-j] (with bounds check)
    So dM/dh_i = sum_j w_j * (h[j-i] if 0<=j-i<N else 0 + h[i-j] if 0<=i-j<N else 0)
    """
    h = np.asarray(h, dtype=float)
    S = np.sum(h)
    if abs(S) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    # Soft max weights
    e = np.exp(-beta * (conv - m))
    w_sum = np.sum(e)
    if w_sum < 1e-300:
        w = np.zeros_like(conv)
        w[np.argmax(conv)] = 1.0
    else:
        w = e / w_sum
    
    M = m + np.log(w_sum) / beta
    
    # dM/dh_i = sum_j w_j * d(conv[j])/d(h[i])
    # d(conv[j])/d(h[i]) = h[j-i] if 0 <= j-i < N, plus h[i-j] if 0 <= i-j < N
    # This is equivalent to: (w * h) circular-like convolution
    # For full conv: dM/dh_i = sum_j w_j * (h[(j-i) % N] if in range) ... 
    # Actually for full (not circular) conv:
    # conv[j] = sum_k h[k]*h[j-k] for valid k
    # d(conv[j])/d(h[i]) = h[j-i] if 0<=j-i<N, plus h[i-j] if 0<=i-j<N
    # So dM/dh_i = sum_j w_j * h[j-i] (if valid) + sum_j w_j * h[i-j] (if valid)
    #             = sum_k h[k] * w[i+k] (if valid) + sum_k h[k] * w[i-k] (if valid)
    
    # Use FFT to compute this efficiently
    # Let W = w padded to size, H = h padded
    # dM/dh_i = sum_k h[k] * w[i+k] + sum_k h[k] * w[i-k]
    # This is like cross-correlation of h and w
    
    fsize = 2 * N
    H_f = np.fft.rfft(h, fsize)
    W_f = np.fft.rfft(w, fsize)
    
    # sum_k h[k] * w[i+k]: this is cross-correlation
    # corr[i] = sum_k h[k] * conj(w[k-i]) ... let me think more carefully
    # sum_k h[k] * w[i+k] for i+k in [0, 2N-2]
    # Let's use: if we define v[j] = w[j], then sum_k h[k]*v[i+k] = (h * v_flipped)[i]
    # where v_flipped[j] = v[-j] ... this is getting complex.
    
    # Simpler: just compute directly for N=400, it's O(N^2) = 160000 ops, fast enough
    dM_dh = np.zeros(N)
    for i in range(N):
        # sum_j w[j] * h[j-i] if 0 <= j-i < N
        # j ranges over [0, 2N-2], j-i in [0, N-1] => j in [i, i+N-1]
        j_start = i
        j_end = min(i + N, 2 * N - 1)
        if j_start <= j_end - 1:
            dM_dh[i] += np.dot(w[j_start:j_end], h[0:j_end - j_start])
        
        # sum_j w[j] * h[i-j] if 0 <= i-j < N
        # i-j in [0, N-1] => j in [i-N+1, i]
        j_start2 = max(0, i - N + 1)
        j_end2 = i + 1
        if j_start2 <= j_end2 - 1:
            # h[i-j] for j in [j_start2, j_end2-1] => h[i-j_end2+1 : i-j_start2+1] reversed
            # h[i - (j_start2 : j_end2)] = h[i-j_end2+1 : i-j_start2+1][::-1]
            idx = i - np.arange(j_start2, j_end2)
            dM_dh[i] += np.dot(w[j_start2:j_end2], h[idx])
    
    # f = 2*N*M / S^2
    # df/dh_i = 2*N * (dM_dh[i] * S^2 - M * 2*S) / S^4
    #         = 2*N * (dM_dh[i] / S^2 - 2*M / S^3)
    val = 2.0 * N * M / (S * S)
    grad = 2.0 * N * (dM_dh / (S * S) - 2.0 * M / (S * S * S))
    
    return val, grad

def _objective_only(h):
    val, _ = _objective_and_grad(h)
    return val

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones(N)
    
    best_h = np.ones(N).astype(float)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Multiple random restarts with different initializations
    n_restarts = 5
    
    for trial in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        # Different initialization strategies
        if trial == 0:
            # Start from ones (seed)
            h0 = np.ones(N)
        elif trial == 1:
            # Random with positive bias
            h0 = rng.normal(1.0, 0.5, N)
        elif trial == 2:
            # Low-frequency dominant
            h0 = np.cos(2 * np.pi * np.arange(N) / N) + 1.0
        elif trial == 3:
            # Random sparse-ish
            h0 = rng.normal(0.5, 1.0, N)
            h0[h0 < 0] *= 0.1
        else:
            # Best so far with perturbation
            h0 = best_h + rng.normal(0, 0.1, N)
        
        # Ensure sum is not zero
        S = np.sum(h0)
        if abs(S) < 1e-10:
            h0 += 1.0
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective_only,
                h0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if np.all(np.isfinite(result.x)):
                res_val, _ = _objective_and_grad(result.x)
                if res_val < best_val:
                    best_val = res_val
                    best_h = result.x.copy()
        except Exception:
            pass
        
        # Also try from the best found so far with smaller perturbation
        if trial > 0:
            h0_pert = best_h + rng.normal(0, 0.05, N)
            try:
                result = minimize(
                    _objective_only,
                    h0_pert,
                    jac=lambda x: _objective_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if np.all(np.isfinite(result.x)):
                    res_val, _ = _objective_and_grad(result.x)
                    if res_val < best_val:
                        best_val = res_val
                        best_h = result.x.copy()
            except Exception:
                pass
    
    # Final polish with more iterations
    if time.time() - START_TIME < TIME_LIMIT - 10:
        try:
            result = minimize(
                _objective_only,
                best_h,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if np.all(np.isfinite(result.x)):
                res_val, _ = _objective_and_grad(result.x)
                if res_val < best_val:
                    best_val = res_val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Validate
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
