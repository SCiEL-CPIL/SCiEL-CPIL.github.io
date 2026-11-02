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

def _soft_max_abs_ac(f, beta=100.0):
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(abs_ac)
    exp_terms = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_max_val = m + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    return soft_max_val, weights

def _soft_score(f, beta=100.0):
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    val, _ = _soft_max_abs_ac(f, beta)
    return val / (s * s)

def _soft_score_grad(f, beta=100.0):
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.zeros_like(f), np.inf
    
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    soft_max_val, weights = _soft_max_abs_ac(f, beta)
    
    if soft_max_val < 1e-15:
        return np.zeros_like(f), 0.0
    
    w = weights  # length 2*n-1
    w_padded = np.zeros(2 * size)
    w_padded[:len(w)] = w
    
    W = np.fft.rfft(w_padded, 2 * size)
    F_full = np.fft.rfft(f, 2 * size)
    
    conv1 = np.fft.irfft(W * F_full, 2 * size)[:2 * n - 1]
    
    w_flip = w[::-1]
    w_flip_padded = np.zeros(2 * size)
    w_flip_padded[:len(w_flip)] = w_flip
    W_flip = np.fft.rfft(w_flip_padded, 2 * size)
    conv2 = np.fft.irfft(W_flip * F_full, 2 * size)[:2 * n - 1]
    
    grad_soft_max = conv1 + conv2
    
    grad = grad_soft_max / (s * s) - 2 * soft_max_val / (s * s * s)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(f)
    
    return grad, soft_max_val / (s * s)

def _structured_init(freq, phase=0.0):
    t = np.arange(N)
    x = np.cos(2 * np.pi * freq * t / N + phase)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def _random_init():
    x = np.random.randn(N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def _banded_init(bandwidth=5):
    x = np.zeros(N)
    for i in range(N):
        for j in range(max(0, i-bandwidth), min(N, i+bandwidth+1)):
            x[j] += np.random.randn()
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def construct():
    deadline = time.time() + 115.0
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    np.random.seed(42)
    
    # Phase 1: Multiple restarts with L-BFGS-B on soft objective
    # Use structured and random initializations
    inits = [np.ones(N)]
    # Add structured initializations at different frequencies
    for freq in [1, 2, 3, 5, 7, 10, 13, 17]:
        inits.append(_structured_init(freq))
    # Add random initializations
    for _ in range(4):
        inits.append(_random_init())
    # Add banded initializations
    for bw in [3, 5, 8]:
        inits.append(_banded_init(bw))
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        def objective_and_grad(x_flat):
            g, val = _soft_score_grad(x_flat, beta=100.0)
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
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 800
                }
            )
            
            x_opt = result.x
            sc = _score(x_opt)
            
            if sc < best_score:
                best_score = sc
                best_x = x_opt.copy()
                
        except Exception:
            pass
    
    # Phase 2: Local refinement with higher beta
    if time.time() < deadline - 10:
        x = best_x.copy()
        
        def objective_and_grad_high(x_flat):
            g, val = _soft_score_grad(x_flat, beta=200.0)
            if not np.all(np.isfinite(g)):
                g = np.zeros_like(x_flat)
            if not np.isfinite(val):
                val = 1e10
            return val, g
        
        try:
            result = minimize(
                objective_and_grad_high,
                x,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 1500
                }
            )
            x_opt = result.x
            sc = _score(x_opt)
            if sc < best_score:
                best_score = sc
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Vectorized peak-targeted coordinate descent
    if time.time() < deadline - 5:
        x = best_x.copy()
        
        # Find the lag with maximum absolute autoconvolution
        ac = _autoconv(x)
        abs_ac = np.abs(ac)
        peak_lag = np.argmax(abs_ac)
        
        step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        
        for step in step_sizes:
            for _ in range(500):
                if time.time() > deadline - 3:
                    break
                
                # Try perturbing random indices
                idx = np.random.randint(N)
                x_old = x[idx]
                
                for direction in [-1, 1]:
                    x[idx] = x_old + direction * step
                    sc = _score(x)
                    if sc < best_score:
                        best_score = sc
                        best_x = x.copy()
                        break
                else:
                    x[idx] = x_old
        
        # Additional: try perturbing indices near the peak lag structure
        if time.time() < deadline - 3:
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Perturb pairs of indices
                i1 = np.random.randint(N)
                i2 = np.random.randint(N)
                x_old1 = x[i1]
                x_old2 = x[i2]
                
                x[i1] += 0.1
                x[i2] -= 0.1
                sc = _score(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
                else:
                    x[i1] = x_old1
                    x[i2] = x_old2
    
    return best_x
