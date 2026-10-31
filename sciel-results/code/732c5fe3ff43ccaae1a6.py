import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 600
dx = 1.0 / (2 * N)

def autoconv_max_ratio(h):
    """Compute max(f*f) / (int f)^2 for step function with heights h."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    H = rfft(h, n=n_fft)
    R = irfft(H * np.conj(H), n=n_fft)[:N]
    max_R = R.max()
    return max_R / (dx * total * total)

def compute_objective_and_grad(x):
    # x is the vector h
    # Normalize h to sum to 1 to fix scale
    h = x / x.sum()
    
    # Compute autoconvolution
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    H = rfft(h, n=n_fft)
    R = irfft(H * np.conj(H), n=n_fft)[:N]
    
    # Smooth max approximation
    # T is the temperature
    T = 10.0
    # Use log-sum-exp for stability
    max_R = R.max()
    log_sum_exp = max_R + np.log(np.sum(np.exp(R - max_R)))
    obj_val = log_sum_exp / T
    
    # Gradient with respect to h
    # d/dh (log sum exp(T R)) = (1/T) * (sum exp(T R_k) * dR_k/dh) / sum exp(T R_k)
    # dR_k/dh_i = h_{i-k} + h_{i+k} (with wrap-around or zero padding logic)
    # Actually, R_k = sum_j h_j h_{j+k}.
    # dR_k/dh_i = sum_j (delta_{j,i} h_{j+k} + h_j delta_{j+k,i})
    # = h_{i+k} + h_{i-k}
    # Note: indices are modulo N? No, the convolution is linear, not circular.
    # The FFT computes circular convolution.
    # We padded to 2N, so the first N values of the inverse FFT correspond to linear convolution lags 0 to N-1.
    # The gradient of the circular convolution is circular.
    # But we only care about the first N lags.
    # However, the "smooth max" is over these N lags.
    # The gradient of R_k with respect to h_i in the circular sense is (circular correlation).
    # Since we are minimizing the peak of the linear convolution, and the circular convolution matches the linear one for lags 0..N-1 if the other part is zero (which it is, due to padding), we can use the circular gradient for these lags.
    
    # weights for the soft max
    w = np.exp(R - max_R)
    w = w / w.sum()
    
    # grad_R_k = circ_corr(h, h)
    # We need sum_k w_k * grad_R_k
    # This is equivalent to circ_corr(h, w) ? No.
    # d/dh_i (sum_k w_k R_k) = sum_k w_k (h_{i+k} + h_{i-k})
    # = (w * h)_i + (w * h)_{-i} ... circular convolution of w and h.
    
    grad_R = irfft(rfft(w, n=n_fft) * np.conj(H), n=n_fft)[:N]
    # Wait, H is fft(h).
    # Circular correlation of h and w is irfft(conj(fft(h)) * fft(w)).
    # Let's check.
    # R_k = sum_j h_j h_{j+k}.
    # This is cross-correlation.
    # Gradient of sum_k w_k R_k w.r.t h_i is sum_k w_k (h_{i+k} + h_{i-k}).
    # This is the circular convolution of w and h, plus w and h reversed?
    # Actually, since w is symmetric (autoconvolution weights), it's just circular correlation.
    
    # Let's just compute it directly for safety if N is small, but N=600 is small enough for O(N^2) if needed?
    # 600^2 = 360,000. That's fast.
    # But we can use FFT.
    
    # grad = circular_convolution(w, h) + circular_convolution(w, h[::-1])?
    # No, h is real.
    # Let's use the property: grad_i = sum_k w_k (h_{i+k} + h_{i-k})
    # This is 2 * real part of circular_correlation(w, h) if w is symmetric?
    # w comes from R, which is symmetric.
    
    grad = irfft(np.conj(H) * rfft(w, n=n_fft), n=n_fft)[:N]
    # This gives sum_k w_k h_{k-i}?
    # Let's verify indices.
    # irfft(conj(H) * W)_i = sum_j h_j w_{i-j} (circular).
    # We want sum_k w_k (h_{i+k} + h_{i-k}).
    # Term 1: sum_k w_k h_{i+k}. Let j = i+k => k = j-i. Sum_j w_{j-i} h_j.
    # This is circular correlation of w and h.
    # Term 2: sum_k w_k h_{i-k}. Let j = i-k => k = i-j. Sum_j w_{i-j} h_j.
    # Since w is symmetric (w_k = w_{-k}), these two terms are equal.
    # So grad = 2 * circular_correlation(w, h).
    
    grad = 2.0 * grad
    
    # Chain rule for normalization h = x / sum(x)
    # This is complicated. Better to optimize x such that h = exp(x) / sum(exp(x)) ?
    # Or just use constraints in scipy.
    
    return obj_val, grad

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # Initial guess: Uniform
    x0 = np.ones(N)
    
    # Define bounds
    bounds = [(0, None) for _ in range(N)]
    
    # Define constraints
    # Sum h = 1
    cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1.0, 'jac': lambda x: np.ones(N)})
    
    # Objective function for minimize
    # We need to minimize the smooth max.
    # Since we have the sum constraint, we can just minimize the smooth max of R.
    # But we must ensure h >= 0.
    
    def fun(x):
        h = x / x.sum()
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        H = rfft(h, n=n_fft)
        R = irfft(H * np.conj(H), n=n_fft)[:N]
        T = 10.0
        max_R = R.max()
        log_sum_exp = max_R + np.log(np.sum(np.exp(R - max_R)))
        return log_sum_exp / T

    def jac(x):
        h = x / x.sum()
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        H = rfft(h, n=n_fft)
        R = irfft(H * np.conj(H), n=n_fft)[:N]
        T = 10.0
        max_R = R.max()
        w = np.exp(R - max_R)
        w = w / w.sum()
        
        # Grad of sum_k w_k R_k w.r.t h
        # grad_h = 2 * circ_corr(w, h)
        grad_h = 2.0 * irfft(np.conj(H) * rfft(w, n=n_fft), n=n_fft)[:N]
        
        # Chain rule for h = x / S
        # d h_i / d x_j = delta_{ij}/S - x_i x_j / S^2
        # grad_x_j = sum_i (d obj / d h_i) * (d h_i / d x_j)
        # = grad_h_j / S - (sum_i grad_h_i h_i) x_j / S^2
        
        S = x.sum()
        h = x / S
        term1 = grad_h / S
        term2 = np.sum(grad_h * h) * h / S
        
        return term1 - term2

    # Run optimization
    best_h = x0.copy()
    best_val = autoconv_max_ratio(best_h)
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=jac, bounds=bounds,
                       constraints=cons, options={'maxiter': 500, 'disp': False})
        
        if res.success and res.fun < best_val:
            # Re-normalize
            h_opt = res.x / res.x.sum()
            val = autoconv_max_ratio(h_opt)
            if val < best_val:
                best_h = h_opt
                best_val = val
    except Exception:
        pass

    # Try a few different temperatures T if time permits
    if time.time() < deadline - 20:
        for T in [5.0, 20.0]:
            def fun_T(x, T=T):
                h = x / x.sum()
                n_fft = 1
                while n_fft < 2 * N:
                    n_fft *= 2
                H = rfft(h, n=n_fft)
                R = irfft(H * np.conj(H), n=n_fft)[:N]
                max_R = R.max()
                log_sum_exp = max_R + np.log(np.sum(np.exp(R - max_R)))
                return log_sum_exp / T

            def jac_T(x, T=T):
                h = x / x.sum()
                n_fft = 1
                while n_fft < 2 * N:
                    n_fft *= 2
                H = rfft(h, n=n_fft)
                R = irfft(H * np.conj(H), n=n_fft)[:N]
                max_R = R.max()
                w = np.exp(R - max_R)
                w = w / w.sum()
                grad_h = 2.0 * irfft(np.conj(H) * rfft(w, n=n_fft), n=n_fft)[:N]
                S = x.sum()
                h_norm = x / S
                term1 = grad_h / S
                term2 = np.sum(grad_h * h_norm) * h_norm / S
                return term1 - term2

            try:
                res = minimize(fun_T, best_h, method='L-BFGS-B', jac=jac_T, bounds=bounds,
                               constraints=cons, options={'maxiter': 200, 'disp': False})
                if res.success:
                    h_opt = res.x / res.x.sum()
                    val = autoconv_max_ratio(h_opt)
                    if val < best_val:
                        best_h = h_opt
                        best_val = val
            except Exception:
                pass

    return list(best_h)
