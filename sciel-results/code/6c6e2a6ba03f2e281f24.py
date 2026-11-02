"""Structural shift: Stochastic Subgradient Descent with Spectral Whitening and Momentum."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

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

def _subgradient_step(x, lr, momentum, noise_scale, rng):
    """
    Perform a stochastic subgradient step.
    1. Compute ac.
    2. Find lag k* with max |ac[k*]|.
    3. Compute gradient of |ac[k*]| w.r.t x.
    4. Update x with momentum and noise.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    k_star = np.argmax(abs_ac)
    
    # Gradient of ac[k] w.r.t x[j] is x[j+k] + x[j-k] (with bounds)
    # We need to construct the gradient vector g such that g[j] = d|ac[k_star]|/dx[j]
    # If ac[k_star] > 0, sign is +1, else -1.
    sign = 1.0 if ac[k_star] > 0 else -.0
    
    # g[j] = sign * (x[j + k_star - (N-1)] + x[j - k_star + (N-1)])
    # Note: ac index m corresponds to lag m - (N-1).
    # Let lag = k_star - (N-1).
    lag = k_star - (N - 1)
    
    g = np.zeros(N)
    # Term 1: x[j + lag]
    # Valid j where j+lag in [0, N-1] => j in [-lag, N-1-lag]
    if lag >= 0:
        j_start = 0
        j_end = N - lag
        g[j_start:j_end] += sign * x[j_start+lag:j_end+lag]
    else:
        j_start = -lag
        j_end = N
        g[j_start:j_end] += sign * x[j_start+lag:j_end+lag]
        
    # Term 2: x[j - lag]
    # Valid j where j-lag in [0, N-1] => j in [lag, N-1+lag]
    if lag >= 0:
        j_start = lag
        j_end = N
        g[j_start:j_end] += sign * x[j_start-lag:j_end-lag]
    else:
        j_start = 0
        j_end = N + lag
        g[j_start:j_end] += sign * x[j_start-lag:j_end-lag]
    
    # Normalize gradient by sum^2 to match objective scaling roughly
    s = np.sum(x)
    if abs(s) > 1e-10:
        g /= (s * s)
        
    # Add noise
    noise = rng.normal(0, noise_scale, size=N)
    
    # Momentum update is handled externally or we just return the update direction
    return g + noise

def _spectral_whiten(x, strength=0.1):
    """
    Project x towards a flatter power spectrum to reduce peak factor.
    """
    L_fft = 1
    while L_fft < 2 * N - 1:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    amp = np.abs(X)
    phase = np.angle(X)
    
    # Target amplitude: flat (white noise)
    # Interpolate current amplitude towards 1.0 (normalized)
    target_amp = np.ones_like(amp)
    # Normalize target to match energy? Or just mix.
    new_amp = (1 - strength) * amp + strength * target_amp
    # Ensure DC component is handled correctly if needed, but whitening DC is fine
    
    new_X = new_amp * np.exp(1j * phase)
    x_new = np.fft.irfft(new_X, n=L_fft)[:N]
    
    # Re-normalize sum to be positive and consistent
    s = np.sum(x_new)
    if s < 0:
        x_new = -x_new
    return x_new

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Initialize with a smooth low-pass signal
    t = np.linspace(0, 1, N)
    x = 1.0 + 0.5 * np.sin(2 * np.pi * t) + 0.2 * np.cos(4 * np.pi * t)
    
    # SGD Parameters
    lr = 0.01
    momentum = 0.9
    noise_scale = 0.05
    v = np.zeros(N) # Momentum velocity
    
    # Main Loop: Stochastic Gradient Descent with Spectral Whitening
    max_iter = 20000
    for i in range(max_iter):
        if _check_deadline():
            break
        
        # 1. Subgradient Step
        g = _subgradient_step(x, lr, momentum, noise_scale, rng)
        
        # Momentum Update
        v = momentum * v + g
        x = x - lr * v
        
        # 2. Periodic Spectral Whitening
        if i % 100 == 0:
            x = _spectral_whiten(x, strength=0.05)
            
        # 3. Re-normalization to prevent drift in scale
        # The objective is scale-invariant, but numerical stability benefits from fixed scale
        s = np.sum(x)
        if abs(s) > 1e-6:
            x = x * (N / s) # Normalize sum to N
        else:
            x = np.ones(N)
            
        # 4. Track Best
        val = _objective_only(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        # 5. Adaptive Learning Rate / Noise
        if i % 500 == 0:
            lr *= 0.95
            noise_scale *= 0.95
            
        # 6. Random Restart / Perturbation if stuck
        if i % 1000 == 0:
            # Try a different initialization if current is bad
            if best_val > 1.0:
                x = 1.0 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 10) * t)
                v = np.zeros(N)
    
    # Final Polish with L-BFGS on the best found
    if not _check_deadline():
        try:
            def obj_func(x):
                return _objective_only(x)
            
            # Use a high beta soft-max for the final polish to get smooth gradient
            beta = 100.0
            def obj_and_grad(x):
                ac = _autoconv_fft(x)
                abs_ac = np.abs(ac)
                m = np.max(abs_ac)
                exps = np.exp(beta * (abs_ac - m))
                soft_val = m + np.log(np.sum(exps)) / beta
                weights = exps / np.sum(exps)
                signs = np.sign(ac)
                signs[ac == 0] = 1.0
                grad_ac = weights * signs
                
                h = np.zeros(2 * N - 1)
                h[N-1 : 2*N-1] = grad_ac[0 : N]
                h[0 : N-1] = grad_ac[2*N-2 : N-1 : -1]
                
                L_corr = 1
                while L_corr < 4 * N:
                    L_corr *= 2
                X_f = np.fft.rfft(x, n=L_corr)
                H_f = np.fft.rfft(h, n=L_corr)
                corr1 = np.fft.irfft(np.conj(H_f) * X_f, n=L_corr)[:N]
                h_rev = h[::-1]
                H_rev_f = np.fft.rfft(h_rev, n=L_corr)
                corr2 = np.fft.irfft(np.conj(H_rev_f) * X_f, n=L_corr)[:N]
                grad_x = corr1 + corr2
                
                s = np.sum(x)
                val = soft_val / (s * s)
                grad = grad_x / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
                return val, grad
            
            result = minimize(obj_and_grad, best_x, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 100, 'gtol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
            
    return best_x
# EVOLVE-BLOCK-END
