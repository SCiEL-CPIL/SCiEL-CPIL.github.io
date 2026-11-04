import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _zadoff_chu_sequence(n, q):
    """
    Generate a Zadoff-Chu-like sequence with continuous chirp parameter q.
    The sequence is x[k] = exp(2j * pi * q * k * (k+1) / (2 * n)) for k = 0..n-1.
    We take the real part to get a real-valued sequence.
    """
    k = np.arange(n)
    phase = 2 * np.pi * q * k * (k + 1) / (2 * n)
    return np.cos(phase)

def _soft_max_abs(c, beta):
    """Compute soft-max of absolute values."""
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    exps = np.exp(beta * (abs_c - m))
    sum_exps = np.sum(exps)
    return m + np.log(sum_exps) / beta

def _objective_smooth(x, beta):
    """Smooth objective: soft-max |c_k| / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    sm = _soft_max_abs(c, beta)
    return sm / (s * s)

def _objective_chirp(params, beta):
    """
    Objective function for chirp parameter optimization.
    params: array of chirp parameters [q1, q2, ...]
    We create a sequence as the sum of multiple Zadoff-Chu sequences with different q values.
    """
    n_params = len(params)
    x = np.zeros(N)
    for i, q in enumerate(params):
        x += _zadoff_chu_sequence(N, q)
    
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    c = _autoconv_lin(x)
    sm = _soft_max_abs(c, beta)
    return sm / (s * s)

def _objective_chirp_grad(params, beta):
    """
    Objective and gradient for chirp parameter optimization.
    Uses numerical differentiation for the gradient.
    """
    n_params = len(params)
    x = np.zeros(N)
    for i, q in enumerate(params):
        x += _zadoff_chu_sequence(N, q)
    
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros(n_params)
    
    c = _autoconv_lin(x)
    sm = _soft_max_abs(c, beta)
    obj = sm / (s * s)
    
    # Numerical gradient
    grad = np.zeros(n_params)
    eps = 1e-7
    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        x_plus = np.zeros(N)
        for j, q in enumerate(params_plus):
            x_plus += _zadoff_chu_sequence(N, q)
        s_plus = np.sum(x_plus)
        if abs(s_plus) < 1e-12:
            grad[i] = 0
        else:
            c_plus = _autoconv_lin(x_plus)
            sm_plus = _soft_max_abs(c_plus, beta)
            obj_plus = sm_plus / (s_plus * s_plus)
            grad[i] = (obj_plus - obj) / eps
    
    return obj, grad

def _optimize_chirp_params(n_params, deadline, beta=50.0):
    """
    Optimize chirp parameters using L-BFGS-B.
    Returns the optimal parameters and the best sequence.
    """
    rng = np.random.default_rng(42)
    
    # Initialize with random chirp parameters in [0.1, 0.9]
    x0 = rng.uniform(0.1, 0.9, size=n_params)
    
    best_params = x0.copy()
    best_val = _objective_chirp(x0, beta)
    
    def obj_func(params):
        if time.time() > deadline:
            return 1e10
        return _objective_chirp(params, beta)
    
    def obj_grad_func(params):
        if time.time() > deadline:
            return 1e10, np.zeros(n_params)
        return _objective_chirp_grad(params, beta)
    
    try:
        result = minimize(
            obj_grad_func, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 200}
        )
        if result.x is not None:
            params_new = result.x
            val = _objective_chirp(params_new, beta)
            if val < best_val:
                best_val = val
                best_params = params_new.copy()
    except Exception:
        pass
    
    # Build the sequence from optimal parameters
    x = np.zeros(N)
    for q in best_params:
        x += _zadoff_chu_sequence(N, q)
    
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    
    return x, best_val

def _refine_sequence(x0, deadline, beta=50.0):
    """
    Refine a sequence using L-BFGS-B on the soft-max objective.
    """
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_std(x)
    
    def obj_func(x):
        if time.time() > deadline:
            return 1e10
        return _objective_smooth(x, beta)
    
    def obj_grad_func(x):
        if time.time() > deadline:
            return 1e10, np.zeros_like(x)
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(x)
        
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c)
        m = np.max(abs_c)
        if m < 1e-15:
            return 0.0, np.zeros_like(x)
        
        exps = np.exp(beta * (abs_c - m))
        sum_exps = np.sum(exps)
        weights = exps / sum_exps
        
        sm = m + np.log(sum_exps) / beta
        obj = sm / (s * s)
        
        g = weights * np.sign(c)
        g_padded = np.zeros(size)
        g_padded[:len(g)] = g
        G = np.fft.rfft(g_padded, size)
        X = np.fft.rfft(x, size)
        dsm_dx = np.fft.irfft(2 * X * G, size)[:n]
        
        grad = (1.0 / (s * s)) * dsm_dx - (2.0 * sm / (s ** 3)) * 1.0
        
        return obj, grad
    
    try:
        result = minimize(
            obj_grad_func, x, method='L-BFGS-B', jac=True,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100}
        )
        if result.x is not None:
            x_new = result.x
            s_new = np.sum(x_new)
            if abs(s_new) > 1e-12:
                x_new = x_new / s_new
            val = _metric_std(x_new)
            if val < best_val:
                best_val = val
                best_x = x_new.copy()
    except Exception:
        pass
    
    return best_x

def construct():
    deadline = time.time() + 140.0  # Safety margin (150s limit, 10s buffer)
    
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Strategy 1: Optimize chirp parameters (structural change)
    # Try different numbers of chirp components
    for n_params in [1, 2, 3, 4]:
        if time.time() > deadline - 10:
            break
        
        x_chirp, val = _optimize_chirp_params(n_params, deadline=deadline - 5, beta=50.0)
        if val < best_val:
            best_val = val
            best_x = x_chirp.copy()
        
        # Refine the chirp-based sequence with full L-BFGS
        if time.time() > deadline - 10:
            break
        x_refined = _refine_sequence(x_chirp, deadline=deadline - 5, beta=50.0)
        val_refined = _metric_std(x_refined)
        if val_refined < best_val:
            best_val = val_refined
            best_x = x_refined.copy()
    
    # Strategy 2: Try some traditional initial candidates and refine
    # Random signs
    for i in range(2):
        if time.time() > deadline - 10:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x)
        if abs(s) < 1e-6:
            x += 1.0
            s = np.sum(x)
        x = x / s
        x_refined = _refine_sequence(x, deadline=deadline - 5, beta=50.0)
        val = _metric_std(x_refined)
        if val < best_val:
            best_val = val
            best_x = x_refined.copy()
    
    # Random Gaussian
    for i in range(2):
        if time.time() > deadline - 10:
            break
        x = rng.normal(0, 1, size=N)
        s = np.sum(x)
        if abs(s) < 1e-6:
            x += 1.0
            s = np.sum(x)
        x = x / s
        x_refined = _refine_sequence(x, deadline=deadline - 5, beta=50.0)
        val = _metric_std(x_refined)
        if val < best_val:
            best_val = val
            best_x = x_refined.copy()
    
    # Strategy 3: Multi-sine with random phases
    for i in range(2):
        if time.time() > deadline - 10:
            break
        x = np.zeros(N)
        for _ in range(4 + i):
            freq = rng.uniform(0.5, N / 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 1.0)
            t = np.arange(N)
            x += amp * np.sin(2 * np.pi * freq * t / N + phase)
        s = np.sum(x)
        if abs(s) < 1e-6:
            x += 1.0
            s = np.sum(x)
        x = x / s
        x_refined = _refine_sequence(x, deadline=deadline - 5, beta=50.0)
        val = _metric_std(x_refined)
        if val < best_val:
            best_val = val
            best_x = x_refined.copy()
    
    # Final: use remaining time for additional random restarts if available
    while time.time() < deadline - 8:
        x_init = rng.normal(0, 1, size=N)
        s = np.sum(x_init)
        if abs(s) < 1e-6:
            x_init += 1.0
            s = np.sum(x_init)
        x_init = x_init / s
        x_opt = _refine_sequence(x_init, deadline=deadline - 3, beta=50.0)
        val = _metric_std(x_opt)
        if val < best_val:
            best_val = val
            best_x = x_opt.copy()
    
    return best_x
