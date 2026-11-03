"""Model-and-optimize: spectral phase retrieval + adaptive L-BFGS + epigraph + subgradient polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

# Pre-compute FFT sizes
L_CONV = 2 * N - 1
L_FFT = 1
while L_FFT < L_CONV:
    L_FFT *= 2

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    X = np.fft.rfft(x, n=L_FFT)
    C = np.fft.irfft(X * X, n=L_FFT)[:L_CONV]
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
    Compute soft-max surrogate for max|ac| and its gradient.
    Uses L_FFT for cross-correlation (sufficient for N-length output).
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Avoid overflow in exp
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Cross-correlation: grad_x[n] = 2 * sum_k grad_ac[k] * x[k-n]
    # Use FFT-based cross-correlation with L_FFT (>= 2N-1, sufficient)
    g_padded = np.zeros(L_FFT)
    g_padded[:L_CONV] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_FFT)
    G_f = np.fft.rfft(g_padded, n=L_FFT)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_FFT)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _epigraph_objective_and_grad(z, beta):
    """
    Epigraph formulation: minimize t s.t. |ac[k]| <= t * s^2 for all k.
    z = [x (N), t (1)]
    Uses log-sum-exp barrier for the constraints.
    """
    x = z[:N]
    t = z[N]
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N + 1)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Barrier: sum_k log(max(0, |ac[k]| - t*s^2 + epsilon)) 
    # Use soft version: sum_k logsumexp(0, |ac[k]| - t*s^2)
    diff = abs_ac - t * s * s
    # logsumexp(0, diff) = log(1 + exp(diff))
    # Stable computation
    max_diff = np.max(diff)
    if max_diff > 50:
        barrier = max_diff + np.log(np.sum(np.exp(diff - max_diff)))
    elif max_diff < -50:
        barrier = np.log(np.sum(np.exp(diff)))
    else:
        barrier = np.log(1.0 + np.sum(np.exp(diff)))
    
    val = t + beta * barrier
    if val > 1e8:
        val = 1e8
        return val, np.zeros(N + 1)
    
    # Gradient
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    
    # Gradient of barrier w.r.t. x
    # d/dx sum_k log(1+exp(|ac[k]| - t*s^2))
    # = sum_k exp(|ac[k]| - t*s^2)/(1+exp(|ac[k]| - t*s^2)) * sign(ac[k]) * d|ac[k]|/dx
    sigmoid_weights = np.exp(diff) / (1.0 + np.exp(diff))
    grad_ac = sigmoid_weights * signs
    
    g_padded = np.zeros(L_FFT)
    g_padded[:L_CONV] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_FFT)
    G_f = np.fft.rfft(g_padded, n=L_FFT)
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_FFT)
    grad_x = 2.0 * beta * corr[:N]
    
    # Gradient w.r.t. t: 1 - beta * sum_k sigmoid_weights * s^2
    grad_t = 1.0 - beta * np.sum(sigmoid_weights) * s * s
    
    grad = np.empty(N + 1)
    grad[:N] = grad_x
    grad[N] = grad_t
    
    return val, grad

def _spectral_init(rng, mode=0):
    """Generate initial x via spectral phase retrieval."""
    n_freq = L_FFT // 2 + 1
    if mode == 0:
        freqs = np.fft.rfftfreq(L_FFT)
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_FFT / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_FFT)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_FFT * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_FFT)
        mag = np.exp(-20 * (freqs * L_FFT / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_FFT)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(n_freq)
    
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_FFT)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _subgradient_polish(x0, rng, max_iters=2000, top_k=5):
    """
    Polish using subgradient of the exact max function.
    The subgradient of max(|ac|) is a convex combination of the gradients of the active peaks.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    # Adaptive step size
    step_size = 0.01
    
    for it in range(max_iters):
        if _check_deadline():
            break
            
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        s = np.sum(best_x)
        
        if abs(s) < 1e-10:
            break
            
        # Find top k peaks
        idx = np.argsort(-abs_ac)[:top_k]
        weights = abs_ac[idx]
        w_sum = np.sum(weights)
        if w_sum < 1e-10:
            break
        weights = weights / w_sum
        
        # Compute subgradient using FFT for speed
        grad_ac = np.zeros(L_CONV)
        for j, k in enumerate(idx):
            sign_k = np.sign(ac[k])
            if sign_k == 0:
                sign_k = 1.0
            grad_ac[k] += 2.0 * weights[j] * sign_k
        
        # Cross-correlation via FFT
        g_padded = np.zeros(L_FFT)
        g_padded[:L_CONV] = grad_ac
        X_f = np.fft.rfft(best_x, n=L_FFT)
        G_f = np.fft.rfft(g_padded, n=L_FFT)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_FFT)
        grad_x = corr[:N]
        
        # Gradient of the full objective J = max|ac| / s^2
        # dJ/dx = (1/s^2) * grad_max - 2 * (max/s^3) * ones
        grad_obj = grad_x / (s * s) - 2.0 * m * np.ones(N) / (s * s * s)
        
        # Normalize gradient
        norm_g = np.linalg.norm(grad_obj)
        if norm_g < 1e-10:
            break
        grad_obj = grad_obj / norm_g
        
        # Line search
        improved = False
        for scale in [step_size, step_size * 0.5, step_size * 2.0, step_size * 0.1]:
            x_trial = best_x - scale * grad_obj
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                improved = True
                break
        
        if improved:
            step_size = max(0.001, step_size * 0.95)
        else:
            step_size = min(0.05, step_size * 1.05)
            
    return best_x, best_val

def _high_precision_polish(x0, rng, max_iters=500):
    """Final high-precision polish with very high beta and tight tolerances."""
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for beta in [2000.0, 5000.0, 10000.0]:
        if _check_deadline():
            break
        
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 150, 'gtol': 1e-13, 
                                                 'ftol': 1e-16, 'maxls': 30})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    # Epigraph polish
    if not _check_deadline():
        try:
            s0 = np.sum(best_x)
            ac0 = _autoconv_fft(best_x)
            t0 = np.max(np.abs(ac0)) / (s0 * s0)
            z0 = np.concatenate([best_x, [t0 * 1.01]])
            
            def epig_func(z, _beta=50.0):
                return _epigraph_objective_and_grad(z, _beta)
            
            result = minimize(epig_func, z0, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-12,
                                                 'ftol': 1e-15, 'maxls': 40})
            
            x_epig = result.x[:N]
            val_epig = _objective_only(x_epig)
            if val_epig < best_val:
                best_val = val_epig
                best_x = x_epig.copy()
        except Exception:
            pass
    
    # Final coordinate descent with very small steps
    step_size = 0.005
    no_improve = 0
    max_no_improve = 2000
    
    while no_improve < max_no_improve and not _check_deadline():
        r = rng.random()
        if r < 0.3:
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.6:
            n_perturb = rng.integers(1, 5)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        else:
            freq = rng.integers(1, 4)
            phase = rng.uniform(0, 2 * np.pi)
            t_arr = np.arange(N)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.01, step_size * 1.02)
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
    ]
    
    # Add diverse spectral initializations (reduced count to save time)
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(6):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Phase 1: L-BFGS with moderate beta (smooth landscape)
    # Process all initial points but with fewer iterations to save time for polishing
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        try:
            def obj_and_grad_func(x, _beta=50.0):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 150, 'gtol': 1e-10, 
                                                 'ftol': 1e-12, 'maxls': 30})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Subgradient polish (more robust than peak-targeted perturbations)
    x_sub, val_sub = _subgradient_polish(best_x, rng, max_iters=1500, top_k=5)
    if val_sub < best_val:
        best_val = val_sub
        best_x = x_sub.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: High-precision polish
    x_hp, val_hp = _high_precision_polish(best_x, rng, max_iters=500)
    if val_hp < best_val:
        best_val = val_hp
        best_x = x_hp.copy()
    
    return best_x
# EVOLVE-BLOCK-END
