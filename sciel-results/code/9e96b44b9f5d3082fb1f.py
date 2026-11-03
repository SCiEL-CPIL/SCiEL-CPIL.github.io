import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N  # step size

def _autoconv(h):
    """Full autoconvolution of h (length 2N-1)."""
    return np.correlate(h, h, mode='full')

def _ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h: array of nonnegative heights.
    Returns the ratio value (higher is better).
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    c = _autoconv(h)
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def _soft_max(c, temp=1.0):
    """
    Compute soft-max of c using log-sum-exp trick.
    Returns the value of soft-max and the gradient w.r.t c.
    """
    c_max = np.max(c)
    exp_c = np.exp((c - c_max) / temp)
    sum_exp = np.sum(exp_c)
    soft_max_val = c_max + temp * np.log(sum_exp)
    grad = exp_c / sum_exp
    return soft_max_val, grad

def _neg_log_ratio_smooth(h, temp=0.1):
    """
    Objective function for minimization: -log(ratio) with soft-max for sup norm.
    We maximize ratio = L2^2 / (L1 * SupC).
    Minimize -log(L2^2) + log(L1) + log(SupC).
    Since h is non-negative, we work with x = log(h) or just h with bounds.
    Let's work with h directly and use bounds [0, inf).
    """
    h = np.maximum(h, 1e-12) # Avoid log(0)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 1e10, np.zeros_like(h)
    
    c = _autoconv(h)
    sup_c, grad_sup_c = _soft_max(c, temp)
    
    # Objective: -log(l2_sq) + log(l1) + log(sup_c)
    # d/dh log(l1) = 1/l1
    # d/dh log(l2_sq) = 2h / l2_sq
    # d/dh log(sup_c) = grad_sup_c * d(c)/dh
    
    # c = h * h (autoconvolution)
    # c[k] = sum_j h[j] h[k-j] (indices adjusted for full correlation)
    # d(c[k])/dh[i] = h[k-i] + h[i-k] (if valid)
    # Note: np.correlate(h, h, 'full') gives c_full.
    # c_full[m] = sum_n h[n] h[n - (m - (N-1))]
    # Let lag = m - (N-1).
    # d(c_full[m])/dh[i] = h[i - lag] + h[i + lag] (with bounds checks)
    
    # We need d(log(sup_c))/dh[i] = sum_k grad_sup_c[k] * d(c_full[k])/dh[i]
    # d(c_full[k])/dh[i] = h[i - (k-(N-1))] + h[i + (k-(N-1))]
    
    # Vectorized calculation of gradient of autoconvolution w.r.t h
    # This is effectively the convolution of grad_sup_c with h, but careful with indices.
    # Let g = grad_sup_c.
    # d(c)/dh[i] = (g * h)[i] + (g * h)[-i]? No.
    # c = h * h.
    # dc/dh[i] is the vector of derivatives of all c[k] w.r.t h[i].
    # c[k] = sum_j h[j] h[k-j].
    # dc[k]/dh[i] = h[k-i] + h[i-k] (if indices valid).
    # We want sum_k g[k] * dc[k]/dh[i].
    # This is the correlation of g and h.
    
    # Let's compute grad_ratio_sup = np.correlate(g, h, mode='full')?
    # np.correlate(a, b, 'full') at index m is sum_n a[n] b[n - (m - (len(a)-1))]
    # Let's just use the property:
    # sum_k g[k] h[k-i] = (g * h)[i] (convolution)
    # sum_k g[k] h[i-k] = (g * h)[i] (since g is symmetric? No, g is not necessarily symmetric, but c is symmetric so g is symmetric).
    # Since h is real, c is symmetric, so g is symmetric.
    # Thus sum_k g[k] (h[k-i] + h[i-k]) = 2 * (g * h)[i].
    
    # Let's verify indices.
    # c_full index m corresponds to lag l = m - (N-1).
    # g is defined on c_full indices.
    # We need conv(g, h).
    
    conv_gh = np.convolve(g, h, mode='full')
    # The term sum_k g[k] h[k-i] corresponds to conv_gh at index i + (N-1)?
    # Let's stick to a simpler numerical gradient check or just use the analytic form carefully.
    
    # Actually, simpler:
    # grad_obj = -2h/l2_sq + 1/l1 + 2 * np.convolve(g, h, mode='same')? 
    # Let's use a safe numerical gradient for the polish step if analytic is tricky, 
    # but analytic is faster.
    
    # Let's re-derive:
    # J = -log(l2_sq) + log(l1) + log(sup_c)
    # dJ/dh = -2h/l2_sq + 1/l1 + d(log(sup_c))/dh
    # d(log(sup_c))/dh = d(sup_c)/sup_c / dh = (1/sup_c) * d(sup_c)/dh
    # d(sup_c)/dh = sum_k g[k] d(c[k])/dh
    # d(c[k])/dh[i] = h[k-i] + h[i-k]
    # sum_k g[k] h[k-i] = (g*h)[i] (convolution)
    # sum_k g[k] h[i-k] = (g*h)[i] (since g is symmetric)
    # So d(sup_c)/dh[i] = 2 * (g*h)[i]
    
    # Note: np.convolve(g, h, 'full') length is 2N-1.
    # We need the part corresponding to indices 0..N-1 of h.
    # The convolution index m corresponds to sum_n g[n] h[m-n].
    # We want sum_k g[k] h[k-i]. Let m = k-i => k = m+i.
    # sum_m g[m+i] h[m]. This is cross-correlation.
    # Since g is symmetric, conv(g, h) = corr(g, h).
    
    # Let's just compute the gradient numerically for reliability in the polish step, 
    # or use the analytic if we can map indices correctly.
    # Given the risk of index errors, I will use a hybrid: 
    # 1. Multiplicative updates (robust, no gradient needed).
    # 2. Simple coordinate descent or numerical gradient polish.
    
    # For the multiplicative update, we don't need the gradient, just the peak locations.
    
    return 1e10, np.zeros_like(h) # Placeholder, we will use a different function for optimization

def _multiplicative_update(h, iters=1000, deadline=None):
    """
    Iterative reweighting to flatten the autoconvolution.
    h_i <- h_i * exp(-eta * (c_peak_contrib_i - mean))
    """
    h = np.maximum(h.copy(), 1e-12)
    h = h / np.sum(h)
    
    for it in range(iters):
        if deadline and time.time() > deadline:
            break
            
        c = _autoconv(h)
        c_max = np.max(c)
        
        # Identify peaks
        # We want to reduce h_i if it contributes to the peak.
        # Contribution of h_i to c[k] is h_i * h[k-i].
        # If c[k] is near max, we should reduce h_i.
        
        # Simple heuristic:
        # grad_approx[i] = sum_{k: c[k] > 0.9*c_max} h[k-i]
        
        # Use soft-max gradient for smoothness
        temp = 0.1
        _, g = _soft_max(c, temp)
        
        # Compute gradient of peak w.r.t h
        # d(c)/dh[i] = sum_k g[k] (h[k-i] + h[i-k])
        # As derived, this is 2 * conv(g, h)[i] (with proper alignment)
        
        # Let's use the property that for symmetric g and h:
        # grad_h[i] = 2 * np.convolve(g, h, mode='full')[i + N - 1] ?
        # Let's check indices.
        # c_full[m] = sum_j h[j] h[j - (m - N + 1)]
        # d(c_full[m])/dh[i] = h[i - (m-N+1)] + h[i + (m-N+1)]
        # Sum_m g[m] h[i - (m-N+1)]
        # Let k = m - N + 1. m = k + N - 1.
        # Sum_k g[k+N-1] h[i-k]
        # This is the cross-correlation of g_full and h at lag i.
        # Since g_full is symmetric, this is conv(g_full, h) at index i + N - 1.
        
        conv_gh = np.convolve(g, h, mode='full')
        grad_peak = 2 * conv_gh[N-1 : N-1+N]
        
        # Update rule: h_i <- h_i * exp(-eta * grad_peak[i])
        # We want to minimize the peak.
        # If grad_peak[i] is high, h_i contributes to the peak, so reduce it.
        
        eta = 0.1
        update = np.exp(-eta * grad_peak)
        
        h = h * update
        h = h / np.sum(h)
        
    return h

def construct():
    """
    Construct a height vector using multiplicative updates and local optimization.
    """
    deadline = time.time() + 115  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Initialize with random positive values
    random.seed(42)
    np.random.seed(42)
    
    # Try a few different initializations
    for trial in range(5):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        h = np.random.rand(N)
        h = h / np.sum(h)
        
        # Multiplicative update to flatten autoconvolution
        h_opt = _multiplicative_update(h, iters=2000, deadline=deadline - 10)
        
        # Local polish using scipy minimize on the smooth surrogate
        # We maximize ratio, so minimize -log(ratio).
        # We use a soft-max surrogate for the sup norm to make it differentiable.
        
        def objective(x):
            x = np.maximum(x, 1e-12)
            l1 = np.sum(x)
            l2_sq = np.sum(x**2)
            if l1 < 1e-15 or l2_sq < 1e-15:
                return 1e10
            c = _autoconv(x)
            sup_c, _ = _soft_max(c, temp=0.01)
            # -log(l2_sq) + log(l1) + log(sup_c)
            return -np.log(l2_sq) + np.log(l1) + np.log(sup_c)
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, h_opt, method='L-BFGS-B', 
                           bounds=[(0, None)]*N, 
                           options={'maxiter': 1000, 'maxfun': 5000})
            if res.success:
                h_polished = np.maximum(res.x, 0)
                h_polished = h_polished / np.sum(h_polished)
                score = _ratio(h_polished)
                if score > best_score:
                    best_score = score
                    best_h = h_polished
        except Exception:
            pass
        
        # Also check the result of multiplicative update directly
        score_mu = _ratio(h_opt)
        if score_mu > best_score:
            best_score = score_mu
            best_h = h_opt

    # Fallback
    if best_score < 1e-10:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
