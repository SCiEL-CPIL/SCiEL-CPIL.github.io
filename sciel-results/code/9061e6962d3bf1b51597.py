"""Model-and-optimize: Robust control / H-infinity style spectral shaping with alternating projections."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 138  # Safety margin before 150s kill

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
    """Compute the raw objective: max|ac| / (sum(x))^2."""
    ac = _autoconv_fft(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Avoid overflow in exp
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    g = weights * signs  # d(soft_max)/d(ac_k)
    
    # Gradient w.r.t. x: grad_x[i] = sum_k g[k] * d(ac[k])/dx[i]
    # d(ac[k])/dx[i] = x[k-i] + x[i-k] (with bounds)
    # grad_x[i] = sum_k g[k]*(x[k-i] + x[i-k])
    # = sum_k g[k]*x[k-i] + sum_k g[k]*x[i-k]
    # Term 1: sum_k g[k]*x[k-i] -> cross-correlation of g and x
    # Term 2: sum_k g[k]*x[i-k] -> convolution of g and x
    
    L_fft = 1
    while L_fft < 4 * N:
        L_fft *= 2
        
    X_f = np.fft.rfft(x, n=L_fft)
    G_f = np.fft.rfft(g, n=L_fft)
    
    # Cross-correlation: ifft(fft(x) * conj(fft(g)))
    # Convolution: ifft(fft(x) * fft(g))
    # Note: We need to handle the indexing carefully.
    # g is defined on [0, 2N-2].
    
    # Let's use the property:
    # ac = x * x
    # d(ac)/dx = 2 * (x * w) where w is related to the subgradient of max|ac|
    # Actually, let's stick to the derived formula:
    # grad_x[i] = sum_k g[k]*(x[k-i] + x[i-k])
    
    # Term A[i] = sum_k g[k] x[k-i]
    # Let j = k-i => k = i+j. Sum_j g[i+j] x[j]
    # This is the cross-correlation of g and x at lag i?
    # R_gx[i] = sum_j g[j+i] x[j] = ifft(fft(g) * conj(fft(x)))[i]
    
    # Term B[i] = sum_k g[k] x[i-k]
    # Let j = i-k => k = i-j. Sum_j g[i-j] x[j]
    # This is the convolution (g * x)[i] = ifft(fft(g) * fft(x))[i]
    
    # So grad_x = ifft(fft(g) * conj(fft(x))) + ifft(fft(g) * fft(x))
    #            = ifft(fft(g) * (conj(fft(x)) + fft(x)))
    # Wait, conj(X) + X is not 2*Re(X) in the frequency domain in a way that helps directly?
    # X[n] + conj(X[n]) = 2*Re(X[n]).
    # So grad_x = ifft(fft(g) * 2*Re(fft(x)))? No, that's only if we are doing element-wise multiplication.
    # Yes, we are.
    
    # Let's verify dimensions.
    # g is length 2N-1. Padded to L_fft.
    # x is length N. Padded to L_fft.
    
    # A = ifft(fft(g) * conj(fft(x)))
    # B = ifft(fft(g) * fft(x))
    # grad_x = (A + B)[:N]
    
    A = np.fft.irfft(G_f * np.conj(X_f), n=L_fft)[:N]
    B = np.fft.irfft(G_f * X_f, n=L_fft)[:N]
    
    grad_x = A + B
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    # d(val)/dx = (grad_soft * s^2 - soft_val * 2s * 1) / s^4
    #           = grad_soft / s^2 - 2 * soft_val / s^3
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng):
    """
    Generate initial points via spectral amplitude shaping.
    """
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
    
    k = np.arange(L_fft // 2 + 1)
    t = k / (L_fft // 2)
    
    windows = [
        np.ones(L_fft // 2 + 1),
        np.exp(-0.5 * (t - 0.5)**2 / (0.15**2)),
        np.exp(-0.5 * (t - 0.5)**2 / (0.25**2)),
        1.0 / (1.0 + (t / 0.3)**2),
        np.sinc(0.5 * t) ** 2,
        np.exp(-t),
        1.0 / (1.0 + t),
    ]
    
    inits = []
    for w in windows:
        w = np.maximum(w, 0)
        phase = rng.uniform(0, 2*np.pi, size=L_fft // 2 + 1)
        spec = w * np.exp(1j * phase)
        spec_full = np.zeros(L_fft, dtype=complex)
        spec_full[:L_fft//2+1] = spec
        spec_full[L_fft//2+1:] = np.conj(spec[1:L_fft//2][::-1])
        x = np.fft.ifft(spec_full).real[:N]
        x = x - np.min(x) + 0.1
        inits.append(x)
    
    t_arr = np.linspace(0, 1, N)
    inits.append(np.ones(N))
    inits.append(1.0 + 0.5 * np.sin(2 * np.pi * t_arr))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * t_arr))
    inits.append(np.exp(t_arr))
    inits.append(np.exp(-t_arr))
    inits.append(1.0 + 0.8 * (t_arr - 0.5)**2)
    inits.append(1.0 + 0.5 * (t_arr - 0.5))
    
    for _ in range(8):
        inits.append(rng.normal(1.0, 0.3, size=N))
        
    for _ in range(4):
        x = np.ones(N) * 0.1
        n_spikes = rng.integers(3, 10)
        idx = rng.choice(N, size=n_spikes, replace=False)
        x[idx] = 1.0
        inits.append(x)
        
    for _ in range(4):
        x = np.ones(N)
        for freq in rng.integers(1, 5, size=3):
            x += 0.3 * np.sin(2 * np.pi * freq * t_arr + rng.uniform(0, 2*np.pi))
        inits.append(x)
        
    return inits

def _spectral_gradient_descent(x, n_iters=100, lr=0.01, beta=50.0):
    """
    Custom gradient descent on the spectral domain to minimize the soft-max objective.
    """
    best_x = x.copy()
    best_val = _objective_only(best_x)
    
    for it in range(n_iters):
        if _check_deadline():
            break
            
        val, grad = _objective_and_grad(best_x, beta)
        if val < best_val:
            best_val = val
            best_x = best_x.copy()
            
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        # Simple gradient step
        x_trial = best_x - lr * grad
        
        # Check improvement
        val_trial = _objective_only(x_trial)
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
            lr *= 1.05  # Increase step size slightly if improving
        else:
            lr *= 0.8   # Decrease step size if not improving
            
        # Keep lr within bounds
        lr = np.clip(lr, 1e-4, 0.1)
        
    return best_x, best_val

def _peak_targeted_polish(x, n_iters=100):
    """
    Identify dominant autoconvolution lags and apply targeted perturbations to reduce them.
    """
    best_x = x.copy()
    best_val = _objective_only(best_x)
    
    for it in range(n_iters):
        if _check_deadline():
            break
            
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        
        # Find the lag with maximum absolute value
        k_max = np.argmax(abs_ac)
        sign_max = np.sign(ac[k_max])
        
        # We want to reduce ac[k_max].
        # ac[k_max] = sum_i x[i] x[k_max-i]
        # To reduce it, we can perturb x[i] in the direction that decreases this sum.
        # d(ac[k_max])/dx[i] = x[k_max-i] + x[i-k_max]
        # So we want to move x[i] opposite to the gradient.
        
        grad_k = np.zeros(N)
        for i in range(N):
            idx1 = k_max - i
            idx2 = i - k_max
            term = 0
            if 0 <= idx1 < N:
                term += best_x[idx1]
            if 0 <= idx2 < N:
                term += best_x[idx2]
            grad_k[i] = term * sign_max
            
        # Normalize gradient
        norm_g = np.linalg.norm(grad_k)
        if norm_g > 1e-10:
            grad_k /= norm_g
            
        # Try a few step sizes
        for step in [0.01, 0.005, 0.02]:
            x_trial = best_x - step * grad_k
            val_trial = _objective_only(x_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial
                break
                
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
        
    rng = np.random.default_rng(12345)
    
    # Phase 1: Multi-start Spectral Gradient Descent
    initial_points = _spectral_init(rng)
    
    for x0 in initial_points:
        if _check_deadline():
            break
            
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        try:
            x_opt, val_opt = _spectral_gradient_descent(x_start, n_iters=100, lr=0.01)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt
        except Exception:
            pass
            
    if _check_deadline():
        return best_x
        
    # Phase 2: Peak Targeted Polish
    best_x, best_val = _peak_targeted_polish(best_x, n_iters=200)
    
    if _check_deadline():
        return best_x
        
    # Phase 3: Final Spectral Gradient Descent
    best_x, best_val = _spectral_gradient_descent(best_x, n_iters=100, lr=0.005)
    
    if _check_deadline():
        return best_x
        
    # Phase 4: Coordinate Descent Polish
    coord_steps = 0
    max_coord_steps = 100000
    while coord_steps < max_coord_steps:
        if _check_deadline():
            break
        improved = False
        for i in range(0, N, max(1, N // 100)):
            if _check_deadline():
                break
            for delta in [0.01, -0.01, 0.005, -0.005]:
                x_trial = best_x.copy()
                x_trial[i] += delta
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                    improved = True
                    break
        coord_steps += 1
        if not improved:
            break
            
    return best_x
# EVOLVE-BLOCK-END
