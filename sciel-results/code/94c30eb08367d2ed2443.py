"""Signed step functions minimizing max|autoconv|/(sum x)^2 using spectral optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _psd_to_signal(psd):
    """
    Convert a PSD (power spectral density) to a time-domain signal.
    PSD[k] = |X[k]|^2 where X is the DFT.
    We reconstruct X from PSD by taking sqrt and assigning phases.
    For minimum phase or zero-phase, we use phase=0.
    """
    psd = np.maximum(psd, 0)
    
    # Make PSD symmetric for real signal
    psd_sym = np.zeros(N)
    if N % 2 == 0:
        psd_sym[0] = psd[0]
        psd_sym[N//2] = psd[N//2]
        for k in range(1, N//2):
            psd_sym[k] = psd[k]
            psd_sym[N-k] = psd[k]
    else:
        psd_sym[0] = psd[0]
        for k in range(1, (N+1)//2):
            psd_sym[k] = psd[k]
            psd_sym[N-k] = psd[k]
    
    X = np.sqrt(psd_sym).astype(complex)
    x = np.fft.ifft(X).real
    return x

def _softmax_objective(x, beta=30.0):
    """
    Smooth surrogate: soft-max of |autoconv| / (sum x)^2 using log-sum-exp.
    Returns (value, gradient).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    # d(soft_max)/dx via chain rule
    # soft_max = max_val + (1/beta)*log(sum(exp(beta*(|ac|-max_val))))
    # d(soft_max)/d(ac_i) = weights[i] * sign(ac_i)
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # d(ac)/dx: ac = x * x (linear conv)
    # d(ac)/dx[j] involves x and shifted x
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    ds_dx = np.ones(N)
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * ds_dx
    
    return val, grad

def _generate_psd_from_params(params):
    """
    Generate a PSD from spectral parameters.
    params[0]: bandwidth parameter (how concentrated at low freq)
    params[1]: tilt
    params[2]: additional shape
    """
    a = np.clip(params[0], 0.5, 5.0)
    tilt = np.clip(params[1], -2.0, 2.0)
    shape = np.clip(params[2], 0.0, 2.0)
    
    k = np.arange(N)
    kf = k / N
    
    psd = np.exp(-a * kf**2)
    psd = psd * np.exp(tilt * kf)
    
    if shape > 0.01:
        psd = psd * (1.0 + shape * 0.5 * np.sin(2*np.pi*kf))
    
    psd = np.maximum(psd, 0)
    return psd

def _psd_objective(params):
    """Objective in PSD parameter space."""
    psd = _generate_psd_from_params(params)
    x = _psd_to_signal(psd)
    return _objective(x)

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: PSD-based optimization with multiple initializations
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, -0.5, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [2.5, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.5, 0.5],
        [3.0, -0.5, 0.5],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_psd_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                psd_opt = _generate_psd_from_params(result.x)
                x_opt = _psd_to_signal(psd_opt)
                best_val = result.fun
                best_x = x_opt
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Direct PSD optimization using gradient-free methods
    X = np.fft.fft(best_x)
    psd_initial = np.abs(X)**2
    if psd_initial[0] > 0:
        psd_initial = psd_initial / psd_initial[0]
    
    def _psd_obj_func(psd):
        x = _psd_to_signal(psd)
        return _objective(x)
    
    try:
        result = minimize(_psd_obj_func, psd_initial, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_val:
            x_opt = _psd_to_signal(result.x)
            best_val = result.fun
            best_x = x_opt
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Time-domain gradient optimization using soft-max surrogate
    def _obj_func(x):
        val, grad = _softmax_objective(x, beta=30.0)
        return val, grad
    
    try:
        result = minimize(_obj_func, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = _objective(result.x)
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Polish with exact objective using BFGS
    try:
        result = minimize(_objective, best_x, method='BFGS', 
                        options={'maxiter': 200, 'gtol': 1e-10})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Random perturbation refinement with fast evaluation
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    max_iters = 100000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9998)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.5, step_size * 1.05)
                if stall_count % 25000 == 0:
                    big_pert = rng.normal(0, step_size * 5, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
    
    # Final polish
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 150, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
