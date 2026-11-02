"""Model-and-optimize: Spectral PSD shaping + Soft-max L-BFGS + Vectorized Polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient using efficient FFT-based cross-correlation.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Numerical stability for soft max
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    
    if sum_exps == 0:
        return m, np.zeros(N)
        
    log_sum = np.log(sum_exps)
    soft_val = m + log_sum / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # Gradient of soft-max w.r.t. ac values
    
    # Compute grad_x using FFT cross-correlation
    # For linear autoconvolution: ac[k] = sum_i x[i] * x[k-i]
    # d(ac[k])/dx[n] = x[k-n] + x[n-k] (with out-of-bounds as 0)
    # So grad_x[n] = sum_k grad_ac[k] * (x[k-n] + x[n-k])
    
    # Use FFT to compute both terms efficiently
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
        
    G_f = np.fft.rfft(grad_ac, n=L_corr)
    X_f = np.fft.rfft(x, n=L_corr)
    
    # Term 1: sum_k grad_ac[k] * x[k-n]
    # This is cross-correlation of grad_ac and x
    corr1 = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    
    # Term 2: sum_k grad_ac[k] * x[n-k]
    # Let j = n-k, so k = n-j
    # sum_j grad_ac[n-j] * x[j]
    # This is cross-correlation of x and reversed grad_ac
    grad_ac_rev = grad_ac[::-1]
    G_rev_f = np.fft.rfft(grad_ac_rev, n=L_corr)
    corr2 = np.fft.irfft(G_rev_f * np.conj(X_f), n=L_corr)
    
    grad_x = corr1[:N] + corr2[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    # grad = (grad_soft * s^2 - soft_val * 2s * 1) / s^4
    #      = grad_soft / s^2 - 2 * soft_val / s^3
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase/amplitude manipulation.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    freqs = np.fft.rfftfreq(L_fft)
    
    if mode == 0:
        # Flat top with taper
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        # Gaussian
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 2:
        # Random positive
        mag = 1.0 + 0.5 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    else:
        mag = np.ones(n_freq)
        
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
        
    return x

def _psd_shaping(x0, iters=50, beta=20.0):
    """
    Optimize x using gradient descent with spectral projection.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    lr = 0.02
    
    for it in range(iters):
        if _check_deadline():
            break
            
        val, grad = _objective_and_grad(best_x, beta)
        
        # Gradient step
        x_trial = best_x - lr * grad
        
        # Normalize to sum=1
        s_trial = np.sum(x_trial)
        if s_trial < 0:
            x_trial = -x_trial
            s_trial = -s_trial
        if s_trial > 1e-6:
            x_trial = x_trial / s_trial
            
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            lr = min(0.1, lr * 1.1)
        else:
            lr *= 0.8
            
    return best_x, best_val

def _coordinate_polish(x, max_steps=3000):
    """
    Fast coordinate descent polish using in-place updates.
    """
    best_x = x.copy()
    best_val = _objective_only(best_x)
    step_size = 0.1
    no_improve = 0
    max_no_improve = 300
    
    rng = np.random.default_rng(123)
    
    for step in range(max_steps):
        if _check_deadline():
            break
        if no_improve > max_no_improve:
            break
            
        improved = False
        # Try random coordinates
        n_try = min(30, N)
        indices = rng.choice(N, size=n_try, replace=False)
        
        for i in indices:
            old_val = best_x[i]
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size, 0.1*step_size, -0.1*step_size]:
                best_x[i] = old_val + delta
                val = _objective_only(best_x)
                if val < best_val:
                    best_val = val
                    improved = True
                    break
                else:
                    best_x[i] = old_val
                    if val > best_val * 1.01:
                        break
        
        if improved:
            no_improve = 0
            step_size = min(0.3, step_size * 1.03)
        else:
            no_improve += 1
            step_size *= 0.92
            
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
        
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral Initialization + Gradient Descent
    for mode in range(3):
        if _check_deadline():
            break
        x0 = _spectral_init(rng, mode)
        x_polished, val_polished = _psd_shaping(x0, iters=30, beta=20.0)
        if val_polished < best_val:
            best_val = val_polished
            best_x = x_polished.copy()
            
    if _check_deadline():
        return best_x
        
    # Phase 2: L-BFGS with annealed beta
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x):
                return _objective_and_grad(x, beta)
                
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
            
    if _check_deadline():
        return best_x
        
    # Phase 3: Coordinate Descent Polish
    best_x, best_val = _coordinate_polish(best_x, max_steps=3000)
    
    return best_x
# EVOLVE-BLOCK-END
