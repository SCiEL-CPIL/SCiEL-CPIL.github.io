"""
Strategy: Direct 400-Dim Coordinate Descent with Smooth Surrogate for Max-Convolution

This strategy fundamentally changes the optimization approach. Instead of using a
low-dimensional spectral parameterization (Fourier coefficients) or structured waveforms
(Zadoff-Chu), we work directly in the full 400-dimensional time domain but use a
coordinate descent method with a smooth surrogate objective.

Key Insights:
1.  **Smooth Surrogate**: The objective involves np.max(np.convolve(h, h)), which is
    non-differentiable. We replace it with a soft-max (log-sum-exp) surrogate that is
    smooth and differentiable. This allows for stable gradient-based coordinate updates.
2.  **Coordinate Descent**: We update one coordinate h[i] at a time. For each coordinate,
    we can compute the exact derivative of the surrogate objective with respect to h[i]
    in O(N) time (using precomputed convolution contributions). This makes each update
    fast and the overall method scalable.
3.  **Direct Representation**: Unlike the Fourier approach, this allows the signal to take
    on any shape, including sharp features or localized structures, which might be
    necessary to achieve a lower peak-to-average power ratio in the autoconvolution.
4.  **Multi-Start**: We perform multiple random restarts to escape local minima.

This is structurally different from the current approach (Fourier + Nelder-Mead) because:
- It operates in the full 400D space, not a low-dimensional subspace.
- It uses coordinate descent with a smooth surrogate, not a global local optimizer like Nelder-Mead.
- It does not impose a smoothness prior via spectral truncation.
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _soft_max(conv, beta):
    """
    Compute the soft-max of the convolution vector.
    beta controls the sharpness. Larger beta -> closer to true max.
    """
    # Shift to avoid overflow
    m = np.max(conv)
    if m > 500:
        return m + np.log(np.sum(np.exp((conv - m) * beta)) / beta) / beta
    else:
        return np.log(np.sum(np.exp(conv * beta)) / beta) / beta

def _surrogate_objective(h, beta):
    """
    Compute the surrogate objective: abs(2*N*soft_max(conv)/sum(h)^2)
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (s * s))

def _grad_surrogate(h, beta):
    """
    Compute the gradient of the surrogate objective with respect to h.
    This is used to guide the coordinate descent step.
    
    Let f(h) = 2*N*soft_max(conv(h)) / (sum(h))^2
    We need df/dh[i].
    
    Let M = soft_max(conv), S = sum(h).
    f = 2*N*M / S^2
    
    df/dh[i] = 2*N * [ (dM/dh[i]) * S^2 - M * 2*S * (dS/dh[i]) ] / S^4
             = 2*N * [ (dM/dh[i]) / S^2 - 2*M * (dS/dh[i]) / S^3 ]
    
    dS/dh[i] = 1
    
    dM/dh[i]: M = log(sum(exp(beta*conv))/beta)/beta = (1/beta) * log(sum(exp(beta*conv))) - log(beta)
    dM/dconv[j] = exp(beta*conv[j]) / sum(exp(beta*conv))
    
    conv = h * h (full convolution, length 2N-1)
    conv[j] = sum_k h[k] h[j-k]
    dconv[j]/dh[i] = h[j-i] + h[i-j]  (with bounds check, effectively h[j-i] if j-i in [0,N-1] else 0, plus symmetric term)
    
    So dM/dh[i] = sum_j (exp(beta*conv[j]) / sum(exp(beta*conv))) * (h[j-i] + h[i-j])
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.zeros(N)
    
    conv = np.convolve(h, h)
    # Compute softmax weights
    # For numerical stability
    m = np.max(conv)
    exp_conv = np.exp(beta * (conv - m))
    sum_exp = np.sum(exp_conv)
    if sum_exp == 0:
        return np.zeros(N)
    w = exp_conv / sum_exp  # w[j] = dM/dconv[j] * beta
    
    # dM/dh[i] = (1/beta) * sum_j w[j] * (h[j-i] + h[i-j])
    # Note: w is defined such that dM/dconv[j] = w[j]/beta ? No, w[j] = exp(beta*conv[j])/sum(exp(beta*conv))
    # dM/dconv[j] = w[j]
    # Wait, M = (1/beta) log(sum(exp(beta conv)))
    # dM/dconv[j] = (1/beta) * (beta exp(beta conv[j]) / sum(exp(beta conv))) = exp(beta conv[j]) / sum(exp(beta conv)) = w[j]
    
    # So dM/dh[i] = sum_j w[j] * (dconv[j]/dh[i])
    # dconv[j]/dh[i] = h[j-i] if 0<=j-i<N else 0, plus h[i-j] if 0<=i-j<N else 0
    # Note that h is real, so this is effectively the cross-correlation of w and h.
    
    # Let's compute g[i] = sum_j w[j] * (h[j-i] + h[i-j])
    # This is the sum of two convolutions:
    # term1[i] = sum_j w[j] h[j-i] = (w * h)[i] ? No, convolution is sum_k w[k] h[i-k].
    # Let's use np.convolve.
    # w * h (full convolution) has length 2N-1.
    # conv_wh = np.convolve(w, h)
    # conv_wh[k] = sum_j w[j] h[k-j]
    # We want sum_j w[j] h[j-i]. Let k = j-i => j = k+i.
    # sum_k w[k+i] h[k]. This is not a standard convolution.
    # It is the correlation.
    # corr_wh[i] = sum_j w[j] h[j-i]
    # We can compute this by convolving w with reversed h, or using fft.
    # Alternatively, note that sum_j w[j] h[j-i] is the same as sum_k w[k+i] h[k].
    
    # Let's just use a loop for clarity first, then optimize if needed.
    # But N=400, so a loop of 400 iterations, each O(N) is 160k ops, which is fine.
    
    g = np.zeros(N)
    # Precompute h_reversed for correlation?
    # corr1[i] = sum_j w[j] h[j-i]
    # corr2[i] = sum_j w[j] h[i-j]
    
    # corr2 is just the convolution of w and h, extracted appropriately.
    # conv_wh = np.convolve(w, h)
    # conv_wh[k] = sum_j w[j] h[k-j]
    # We want sum_j w[j] h[i-j]. This is conv_wh[i].
    
    conv_wh = np.convolve(w, h)
    # conv_wh has length 2N-1. Indices 0 to 2N-2.
    # For i in 0..N-1, conv_wh[i] is valid.
    
    # Now corr1[i] = sum_j w[j] h[j-i]
    # Let k = j-i => j = k+i.
    # sum_k w[k+i] h[k]
    # This is the correlation of w and h.
    # We can compute it by convolving w with reversed h?
    # rev_h = h[::-1]
    # conv_w_revh = np.convolve(w, rev_h)
    # conv_w_revh[k] = sum_j w[j] rev_h[k-j] = sum_j w[j] h[N-1-(k-j)] = sum_j w[j] h[N-1-k+j]
    # Let m = N-1-k+j => j = m - (N-1-k)
    # sum_m w[m - (N-1-k)] h[m]
    # This is getting messy. Let's just use a direct loop for corr1, it's small enough.
    
    for i in range(N):
        # corr1[i] = sum_j w[j] h[j-i]
        # j ranges such that j-i is in [0, N-1] => j in [i, i+N-1]
        # w has length 2N-1.
        # h[j-i] is valid if j-i in [0, N-1].
        
        # We can compute this as a dot product of w[i : i+N] and h[0 : N]
        # w_slice = w[i : i+N]
        # h_slice = h[0 : N]
        # corr1_val = np.dot(w_slice, h_slice)
        
        # And corr2[i] = conv_wh[i]
        
        corr1_val = np.dot(w[i : i+N], h)
        corr2_val = conv_wh[i]
        
        g[i] = corr1_val + corr2_val
    
    dM_dh = g  # Since dM/dconv[j] = w[j], and dM/dh[i] = sum_j w[j] dconv[j]/dh[i] = g[i]
    
    # Now compute gradient of f
    # f = 2*N*M / S^2
    # df/dh[i] = 2*N * (dM_dh[i] * S^2 - M * 2*S * 1) / S^4
    #           = 2*N * (dM_dh[i] / S^2 - 2*M / S^3)
    
    M = _soft_max(conv, beta)
    S = s
    grad = 2.0 * N * (dM_dh / (S * S) - 2.0 * M / (S * S * S))
    
    # The objective is abs(f). If f is negative, the gradient sign flips.
    # f = 2*N*M/S^2. M is always positive (soft max of real numbers). S^2 is positive.
    # So f is always positive. abs(f) = f.
    # So the gradient is just grad.
    
    return grad

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(123)
    
    # Parameters for coordinate descent
    beta = 50.0  # Sharpness of soft-max
    max_iter = 200  # Max iterations per restart
    tol = 1e-6
    
    # Multi-start
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize h
        if r == 0:
            h = np.ones(N)
        else:
            # Random initialization with some structure
            # Try a low-pass filtered noise
            h = rng.normal(0, 1, N)
            # Apply a simple low-pass filter (moving average)
            kernel = np.ones(5) / 5
            h = np.convolve(h, kernel, mode='same')
            # Ensure sum is not zero
            s = np.sum(h)
            if abs(s) < 1e-3:
                h += 0.1
            # Normalize to have sum 1 for stability
            h = h / np.sum(h)
        
        current_val = _evaluate(h)
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
        
        # Coordinate descent
        prev_val = _surrogate_objective(h, beta)
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient for coordinate i
                grad = _grad_surrogate(h, beta)
                if not np.isfinite(grad[i]):
                    continue
                
                # Line search: try a few step sizes
                # h_new = h + step * grad[i] * e_i
                # We want to minimize the surrogate objective.
                # Since we are doing coordinate descent, we can try a few steps.
                
                base_h = h.copy()
                base_val = _surrogate_objective(h, beta)
                
                # Try steps: -0.1, -0.01, 0.01, 0.1, 0.5, 1.0 scaled by gradient
                # Gradient magnitude can vary, so scale by 1/max(1, |grad|)
                scale = 1.0 / max(1.0, abs(grad[i]))
                steps = [-0.5 * scale, -0.1 * scale, -0.01 * scale, 0.01 * scale, 0.1 * scale, 0.5 * scale]
                
                best_step = 0.0
                best_new_val = base_val
                
                for step in steps:
                    h_new = base_h.copy()
                    h_new[i] += step
                    new_val = _surrogate_objective(h_new, beta)
                    if new_val < best_new_val:
                        best_new_val = new_val
                        best_step = step
                
                if best_step != 0.0:
                    h[i] += best_step
                    improved = True
            
            if not improved:
                break
            
            # Check convergence
            curr_val = _surrogate_objective(h, beta)
            if abs(prev_val - curr_val) < tol:
                prev_val = curr_val
                break
            prev_val = curr_val
        
        # Update best
        final_val = _evaluate(h)
        if final_val < best_val:
            best_val = final_val
            best_h = h.copy()
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)
        
    return best_h
