"""
Optimized C3-prime task: minimize max(|autoconv|) / (sum(x))^2.

Strategy:
1. Spectral parameterization (low-dimensional) + differential evolution for global search.
2. Continuous gradient descent polish using a smooth surrogate (log-sum-exp) for the
   max-abs autoconvolution term.
3. Discrete bit-flip refinement (binary sign flip) to exploit the signed step structure.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    if n == 0:
        return np.array([])
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]


def _score_from_signal(x):
    """
    Compute the metric: max(|autoconv|) / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (integral * integral)


def _param_to_signal(params):
    """
    Convert spectral coefficients to a time-domain signal.
    params: array of length 2*M+1, representing [c0, c1, d1, c2, d2, ..., cM, dM]
    x[n] = c0 + sum_{k=1}^{M} c_k * cos(2*pi*k*n/N) + d_k * sin(2*pi*k*n/N)
    """
    M = len(params) // 2
    c0 = params[0]
    x = np.full(N, c0)
    n = np.arange(N)
    for k in range(1, M + 1):
        c_k = params[2 * k - 1]
        d_k = params[2 * k]
        x += c_k * np.cos(2 * np.pi * k * n / N) + d_k * np.sin(2 * np.pi * k * n / N)
    return x


def _soft_max_abs_conv(x, beta=30.0):
    """
    Smooth surrogate for max(|autoconv|) using log-sum-exp.
    This is differentiable and suitable for gradient descent.
    """
    conv = _autoconv_full(x)
    abs_c = np.abs(conv)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    # log-sum-exp trick for numerical stability
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max


def _smooth_objective(x):
    """
    Smooth objective: soft_max_abs_conv(x) / (sum(x))^2
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e12
    return _soft_max_abs_conv(x) / (s * s)


def _grad_smooth_objective(x):
    """
    Numerical gradient of the smooth objective using central differences.
    """
    eps = 1e-6
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = _smooth_objective(x_plus)
        f_minus = _smooth_objective(x_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _gradient_descent_polish(x_init, deadline, lr_init=0.01, max_iter=2000, beta=30.0):
    """
    Gradient descent on the smooth surrogate to polish the signal.
    """
    x = x_init.astype(np.float64).copy()
    current_val = _smooth_objective(x)
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        grad = _grad_smooth_objective(x)
        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            break
        
        x_new = x - lr * grad
        new_val = _smooth_objective(x_new)
        
        if new_val < current_val:
            x = x_new
            current_val = new_val
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
    
    return x


def _binary_flip_refine(x_init, deadline, max_passes=10):
    """
    Refine by flipping individual elements between +1 and -1 (or their signs).
    This exploits the 'signed step' nature of the problem.
    """
    x = x_init.copy()
    current_score = _score_from_signal(x)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = x[i]
            # Flip sign
            x[i] = -old_val
            new_score = _score_from_signal(x)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x[i] = old_val
        
        if not improved:
            break
    
    return x, current_score


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Initial best: all ones
    best_x = np.ones(N)
    best_score = _score_from_signal(best_x)
    
    # Differential evolution with spectral parameterization
    M = 15
    n_params = 2 * M + 1
    bounds = [(-10, 10)] * n_params
    
    def _objective(params):
        x = _param_to_signal(params)
        return _score_from_signal(x)
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-8,
            seed=42,
            polish=True,
            updating='immediate'
        )
        
        if result.fun < best_score:
            x_opt = _param_to_signal(result.x)
            score_opt = _score_from_signal(x_opt)
            if score_opt < best_score:
                best_x = x_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_x
    
    # Gradient descent polish on the best signal found
    try:
        x_polished = _gradient_descent_polish(best_x, deadline - 10, lr_init=0.01, max_iter=1000)
        score_polished = _score_from_signal(x_polished)
        if score_polished < best_score:
            best_x = x_polished
            best_score = score_polished
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20:
        return best_x
    
    # Binary flip refinement
    try:
        x_flipped, score_flipped = _binary_flip_refine(best_x, deadline - 5, max_passes=5)
        if score_flipped < best_score:
            best_x = x_flipped
            best_score = score_flipped
    except Exception:
        pass
    
    return best_x

# EVOLVE-BLOCK-END
