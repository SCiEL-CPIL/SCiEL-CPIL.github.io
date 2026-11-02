import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(f):
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)[:2 * n - 1]
    return ac

def _score(f):
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_ac(f, beta=50.0):
    """Compute soft approximation of max|ac| using log-sum-exp."""
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    # Use log-sum-exp for numerical stability
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    # soft max: (1/beta) * log(sum(exp(beta * x)))
    # For stability: m + (1/beta)*log(sum(exp(beta*(x-m))))
    exp_terms = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exp_terms))
    return m + log_sum / beta

def _soft_score(f, beta=50.0):
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return _soft_max_abs_ac(f, beta) / (s * s)

def _soft_score_grad(f, beta=50.0):
    """Compute gradient of soft score w.r.t. f."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.zeros_like(f), np.inf
    
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return np.zeros_like(f), 0.0
    
    # Soft max value
    exp_terms = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_max_val = m + log_sum / beta
    
    # Gradient of soft max w.r.t. abs_ac
    # d(soft_max)/d(abs_ac_i) = exp(beta*(abs_ac_i - m)) / sum(exp(beta*(abs_ac_j - m)))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of soft_max_val w.r.t. f
    # ac = f * f (autoconvolution)
    # d(ac_k)/d(f_i) = f_{k-i} + f_{i-k} (for linear autoconv)
    # But we need d(soft_max)/d(f_i) = sum_k weights_k * d(ac_k)/d(f_i) * sign(ac_k)
    
    # For efficiency, compute via FFT
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    F = np.fft.rfft(f, size)
    
    # d(ac)/d(f) can be computed as convolution with f
    # ac_k = sum_i f_i * f_{k-i}
    # d(ac_k)/d(f_j) = f_{k-j} + f_{j-k} (if both in range)
    
    # Using FFT: gradient of ac w.r.t. f is related to cross-correlation
    # d(ac)/d(f_j) = f * f (cross-correlation at various lags)
    
    # More precisely: ac = ifft(fft(f)^2)
    # d(ac)/d(f) involves the derivative of ifft(fft(f)^2)
    # This is complex; let's use a simpler approach
    
    # Alternative: use the fact that for real f, 
    # d(ac_k)/d(f_j) = f_{k-j} + f_{j-k} where defined
    
    # Let's compute the gradient numerically via FFT
    # The gradient of ac w.r.t. f is:
    # grad_ac[j] = sum_k weights_k * (f_{k-j} + f_{j-k})
    
    # This can be computed as a convolution
    # Let w = weights (length 2*n-1)
    # grad_ac = convolve(w, f) + convolve(w, f[::-1]) (approximately)
    
    # For efficiency, use FFT-based convolution
    w = weights  # length 2*n-1
    w_padded = np.zeros(2*size)
    w_padded[:len(w)] = w
    
    W = np.fft.rfft(w_padded, 2*size)
    F_full = np.fft.rfft(f, 2*size)
    
    # conv(w, f)
    conv1 = np.fft.irfft(W * F_full, 2*size)[:2*n-1]
    
    # conv(w, f_reversed) - but this is trickier
    # Actually, for autoconvolution, the gradient is:
    # d(ac)/d(f_j) = sum_k (f_{k-j} + f_{j-k}) * w_k
    # = (f * w)_j + (f * w_flipped)_j
    
    w_flip = w[::-1]
    w_flip_padded = np.zeros(2*size)
    w_flip_padded[:len(w_flip)] = w_flip
    W_flip = np.fft.rfft(w_flip_padded, 2*size)
    conv2 = np.fft.irfft(W_flip * F_full, 2*size)[:2*n-1]
    
    grad_soft_max = conv1 + conv2
    
    # Gradient of soft_score = soft_max / s^2
    # d(soft_score)/d(f_j) = (d(soft_max)/d(f_j) * s^2 - soft_max * 2*s * 1) / s^4
    # = grad_soft_max[j] / s^2 - 2 * soft_max / s^3
    
    grad = grad_soft_max / (s * s) - 2 * soft_max_val / (s * s * s)
    
    return grad, soft_score(f, beta)

def construct():
    deadline = time.time() + 115.0
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    np.random.seed(42)
    
    # Multiple random restarts with L-BFGS-B optimization
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random sequence
        if restart == 0:
            x0 = np.ones(N)
        else:
            x0 = np.random.randn(N)
            s0 = np.sum(x0)
            if abs(s0) < 1e-6:
                x0 += 0.1
                s0 = np.sum(x0)
            x0 /= s0  # Normalize sum to 1
        
        # Define objective and gradient for L-BFGS-B
        def objective_and_grad(x_flat):
            g, val = _soft_score_grad(x_flat, beta=50.0)
            # Ensure gradient is finite
            if not np.all(np.isfinite(g)):
                g = np.zeros_like(x_flat)
            if not np.isfinite(val):
                val = 1e10
            return val, g
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 500
                }
            )
            
            x_opt = result.x
            sc = _score(x_opt)
            
            if sc < best_score:
                best_score = sc
                best_x = x_opt.copy()
                
        except Exception:
            pass
    
    # Final polish: coordinate descent on the best solution
    if time.time() < deadline - 5:
        x = best_x.copy()
        step_sizes = [0.1, 0.05, 0.01, 0.005]
        
        for step in step_sizes:
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                idx = np.random.randint(N)
                x_old = x[idx]
                x[idx] += np.random.choice([-1, 1]) * step
                sc = _score(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
                else:
                    x[idx] = x_old
    
    return best_x
