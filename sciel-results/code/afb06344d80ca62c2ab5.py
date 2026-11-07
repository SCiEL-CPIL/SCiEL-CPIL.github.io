import numpy as np
import time
from scipy.optimize import minimize

N = 100

def get_r(y):
    return np.correlate(y, y, mode='full')

def objective_and_grad(y, tau, lam):
    # y shape (N,)
    s = np.sum(y)
    # Penalty
    pen = lam * s * s
    grad_pen = 2 * lam * s * np.ones(N)
    
    r = get_r(y)
    c = 25 - r
    
    # Soft max weights
    c_max = np.max(c)
    exp_c = np.exp((c - c_max) / tau)
    sum_exp = np.sum(exp_c)
    w = exp_c / sum_exp
    
    val = c_max + tau * np.log(sum_exp) + pen
    
    # Gradient of soft max
    # d/dy (sum_k w_k c_k) = sum_k w_k dc_k/dy
    # dc_k/dy = - dr_k/dy
    # dr_k/dy[j] = y[j+k] + y[j-k]
    
    # We need sum_k w_k (shift_k(y) + shift_-k(y))
    # This is equivalent to convolving w with y?
    # Let's compute it directly or via FFT?
    # N=100, direct loop is fine.
    
    grad_sm = np.zeros(N)
    # w is length 2N-1. Center is N-1.
    # r_k corresponds to index N-1+k in r?
    # np.correlate(y, y, 'full') returns c[0] for lag -(N-1) ... c[2N-2] for lag N-1.
    # So r_k is at index N-1+k.
    # w is aligned with r.
    # w[i] corresponds to lag i - (N-1).
    
    for k_idx in range(len(w)):
        if w[k_idx] < 1e-12: continue
        k = k_idx - (N - 1)
        # dr_k/dy[j] = y[j+k] + y[j-k]
        # We need to add w[k_idx] * (y[j+k] + y[j-k]) to grad_sm[j]
        # This is w[k_idx] * (shifted y by -k + shifted y by k)
        
        # Efficient way:
        # grad_sm += w[k_idx] * (np.roll(y, -k) + np.roll(y, k)) 
        # But roll wraps around. We need zero padding.
        # For N=100, let's just do it simply or use conv.
        
        # Actually, sum_k w_k S_k y is the convolution of w and y?
        # Let's just use a simple loop for safety, N is small.
        pass

    # Simpler analytic grad:
    # grad_c_k = - (y shifted by k + y shifted by -k)
    # Total grad = - sum_k w_k grad_c_k = sum_k w_k (y shifted by k + y shifted by -k)
    
    # Let's compute H = sum_k w_k S_k. Then grad = H y + H^T y?
    # Since w is symmetric (c is symmetric, r is symmetric), w is symmetric.
    # So grad = 2 * (w * y) where * is convolution?
    # w is the kernel.
    # grad_sm = 2 * np.convolve(y, w, mode='same') ?
    # Let's verify.
    # (w * y)_j = sum_k w_k y_{j-k}.
    # We want sum_k w_k y_{j+k}.
    # Since w is symmetric, w_k = w_{-k}.
    # sum_k w_k y_{j+k} = sum_m w_{-m} y_{j-m} = sum_m w_m y_{j-m} = (w*y)_j.
    # So sum_k w_k y_{j+k} = (w*y)_j.
    # And sum_k w_k y_{j-k} = (w*y)_j.
    # So grad_sm = 2 * np.convolve(y, w, mode='same').
    
    # Note: np.convolve handles boundaries correctly (zero padding).
    # w length 2N-1. y length N.
    # Result length 3N-2? No, 'same' returns length max(M,N).
    # We need the part corresponding to j=0..N-1.
    
    conv_full = np.convolve(y, w, mode='full')
    # The center of conv_full corresponds to j=0?
    # Convolution index: (w*y)_j = sum_k w_k y_{j-k}.
    # In full mode, index m corresponds to sum_k w_k y_{m-k} with appropriate shifts.
    # It's easier to just use the property that grad = 2 * (w * y) centered.
    
    # Let's just compute the gradient numerically to be safe, it's only 100 dims.
    # But analytic is faster.
    
    # Let's use the convolution trick.
    # w is symmetric.
    # grad_sm = 2 * np.convolve(y, w, mode='same') 
    # Is 'same' aligned correctly?
    # np.convolve(a, v, 'same') returns output of length max(M,N).
    # If len(w) > len(y), it returns len(w).
    # We want the central part of length N.
    
    # Let's stick to numerical gradient for correctness, but optimize the inner loop.
    # Or just use L-BFGS-B with numerical gradient, it's robust.
    
    return val, None # Let's use numerical grad in the wrapper

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    i = np.arange(N)
    y0 = np.sin(2 * np.pi * i / 400)
    y0 -= np.mean(y0)
    y0 = y0 / np.max(np.abs(y0)) * 0.49
    y0 = np.clip(y0, -0.5, 0.5)
    y0 -= np.mean(y0)
    max_abs = np.max(np.abs(y0))
    if max_abs > 0.5:
        y0 = y0 / max_abs * 0.5
        
    y = y0.copy()
    best_y = y.copy()
    best_val = value_from_y(best_y)
    
    taus = [10.0, 5.0, 1.0, 0.5, 0.1]
    lams = [100.0, 100.0, 1000.0, 1000.0, 10000.0]
    
    for tau, lam in zip(taus, lams):
        if time.time() > deadline - 10:
            break
            
        def obj_func(y_flat):
            return objective_and_grad(y_flat, tau, lam)[0]
            
        try:
            res = minimize(
                obj_func, 
                y, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * N,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.success:
                y_new = res.x
                y_new -= np.mean(y_new)
                y_new = np.clip(y_new, -0.5, 0.5)
                y_new -= np.mean(y_new)
                max_abs = np.max(np.abs(y_new))
                if max_abs > 0.5:
                    y_new = y_new / max_abs * 0.5
                
                val_new = value_from_y(y_new)
                if val_new < best_val:
                    best_val = val_new
                    best_y = y_new.copy()
                y = y_new
        except Exception:
            pass
            
    h = 0.5 + best_y
    h -= np.mean(h) - 0.5
    h = np.clip(h, 0.0, 1.0)
    
    return h

def value_from_y(y):
    r = np.correlate(y, y, mode='full')
    min_r = np.min(r)
    max_c = 25 - min_r
    return max_c / N * 2
