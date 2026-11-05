import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)
    return C[:2*N-1]

def _objective_and_grad(h, k=50.0):
    """
    Compute objective J(h) = soft_max(c(h)) / (sum(h))^2
    and its gradient.
    """
    s = np.sum(h)
    if s == 0:
        return np.inf, np.zeros_like(h)
    
    c = _autoconv_fft(h)
    
    # Soft max
    m = np.max(c)
    # Avoid overflow
    exp_k_c = np.exp(k * (c - m))
    sum_exp = np.sum(exp_k_c)
    M = m + np.log(sum_exp) / k
    
    # Weights for soft max gradient
    w = exp_k_c / sum_exp
    
    # Gradient of M w.r.t h
    # grad_M = 2 * cross_corr(w, h)
    # cross_corr(w, h)[l] = sum_j w_j h[l-j]
    # Use FFT
    size = 2 * N
    W = np.fft.rfft(w, n=size)
    H = np.fft.rfft(h, n=size)
    # Note: cross_corr(w, h) is ifft(W * conj(H))
    # Let's verify.
    # ifft(W * H) is conv(w, h).
    # ifft(W * conj(H)) is cross_corr.
    cross_corr = np.fft.irfft(W * np.conj(H), n=size)
    grad_M = 2.0 * cross_corr[:N]
    
    # Objective
    J = M / (s * s)
    
    # Gradient of J
    # J = M * s^-2
    # dJ = dM * s^-2 + M * (-2) s^-3 * ds
    # ds = sum(dh) = 1 (vector of ones)
    grad_J = grad_M / (s * s) - 2.0 * M * np.ones(N) / (s * s * s)
    
    return J, grad_J

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.RandomState(42)
    
    # Try several random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize h
        # Mix of random and structured
        if restart % 2 == 0:
            h = rng.randn(N)
        else:
            # Start with alternating signs to reduce center peak
            h = np.ones(N)
            h[::2] = -1.0
            h += 0.1 * rng.randn(N)
            
        # Normalize sum to 1 to avoid issues with s->0
        s = np.sum(h)
        if s != 0:
            h = h / s
        
        # Optimize
        def func(x):
            val, grad = _objective_and_grad(x, k=50.0)
            if not np.isfinite(val):
                return 1e10, np.zeros_like(x)
            return val, grad
        
        try:
            res = scipy.optimize.minimize(func, h, method='L-BFGS-B', 
                                          jac=True, options={'maxiter': 200, 'gtol': 1e-8})
            h_opt = res.x
            # Ensure finite
            if not np.all(np.isfinite(h_opt)):
                continue
                
            # Evaluate exact value
            val = _evaluate(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
            
    # Final polish with coordinate descent?
    # Or just return best_h
    
    return best_h

def _evaluate(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv)) / (s * s)
