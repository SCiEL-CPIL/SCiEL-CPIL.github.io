"""Optimized program for the second autocorrelation inequality task using softmax parameterization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _autoconvolution_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _l2_norm_sq(h):
    """Squared L2 norm."""
    return np.sum(h**2)

def _l1_norm(h):
    """L1 norm."""
    return np.sum(h)

def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    l2sq = _l2_norm_sq(h)
    l1 = _l1_norm(h)
    peak = _autoconvolution_peak(h)
    if l1 == 0 or peak == 0:
        return 0.0
    return l2sq / (l1 * peak)

def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_softmax(y):
    """
    Objective to minimize: negative of the ratio.
    y is a real vector of length N.
    h = softmax(y).
    """
    # Softmax
    y_max = np.max(y)
    exp_y = np.exp(y - y_max)
    h = exp_y / np.sum(exp_y)
    
    l2sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=0.01)
    
    if peak == 0:
        return 0.0
    
    # Since sum(h) = 1, ratio is l2sq / peak
    return -l2sq / peak

def _optimize_softmax(y0, time_limit):
    """Run L-BFGS-B optimization on softmax parameters."""
    start_time = time.time()
    
    result = minimize(
        _objective_softmax,
        y0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # Return h corresponding to initial y0 if timeout
        y_max = np.max(y0)
        exp_y = np.exp(y0 - y_max)
        h = exp_y / np.sum(exp_y)
        return h
    
    y_opt = result.x
    y_max = np.max(y_opt)
    exp_y = np.exp(y_opt - y_max)
    h_opt = exp_y / np.sum(exp_y)
    return h_opt

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    # Initial best: Uniform
    best_h = np.ones(N) / N
    best_score = _ratio(best_h)
    
    initializations_y = []
    
    # 1. Uniform y (leads to uniform h)
    initializations_y.append(np.zeros(N))
    
    # 2. Gaussian-like y
    x = np.linspace(-1, 1, N)
    for width in [0.1, 0.5, 1.0, 2.0]:
        initializations_y.append(-x**2 / width)
        
    # 3. Bimodal y
    for offset in [0.4, 0.6, 0.8]:
        initializations_y.append(-((x - offset)**2) / 0.1 - ((x + offset)**2) / 0.1)
        
    # 4. Random y
    for seed in range(1, 21):
        rng_local = np.random.RandomState(seed)
        initializations_y.append(rng_local.randn(N))
        
    # 5. Structured Random (Smooth)
    for seed in range(21, 31):
        rng_local = np.random.RandomState(seed)
        y_raw = rng_local.randn(N)
        kernel = np.ones(5) / 5
        y_smooth = np.convolve(y_raw, kernel, mode='same')
        initializations_y.append(y_smooth)

    time_per_opt = 3.0 # Seconds per optimization attempt

    for i, y0 in enumerate(initializations_y):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
            
        h_opt = _optimize_softmax(y0, time_per_opt)
        
        score = _ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt

    # Final polish
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        # Convert best_h to y
        y_best = np.log(np.maximum(best_h, 1e-10))
        
        # Run a very tight optimization
        result = minimize(
            _objective_softmax,
            y_best,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 5000}
        )
        
        y_opt = result.x
        y_max = np.max(y_opt)
        exp_y = np.exp(y_opt - y_max)
        h_opt = exp_y / np.sum(exp_y)
        
        score = _ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
            
    return list(best_h)
