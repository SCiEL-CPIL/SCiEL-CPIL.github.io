"""
Alternating minimization with spectral magnitude constraint for signed step functions.
Minimizes max|autoconv|/(sum x)^2 by alternating between time-domain optimization
and spectral projection to a flat-top magnitude profile.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

# Precompute FFT sizes
_L_fft = 1
while _L_fft < 2 * N - 1:
    _L_fft *= 2

_L_fft_grad = 1
while _L_fft_grad < 3 * N:
    _L_fft_grad *= 2

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    X = np.fft.rfft(x, n=_L_fft)
    C = np.fft.irfft(X * X, n=_L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad_smooth(x, beta=30.0):
    """
    Compute soft-max surrogate objective and its gradient.
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
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    G = np.fft.rfft(g, n=_L_fft_grad)
    Xf = np.fft.rfft(x, n=_L_fft_grad)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=_L_fft_grad)[:N]
    term2 = np.fft.irfft(G * Xf, n=_L_fft_grad)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _spectral_projection(x, target_mag=None, strength=0.5):
    """
    Project x onto the set of signals with approximately flat-top magnitude spectrum.
    This enforces a specific spectral shape that is known to be good for minimizing
    peak autoconvolution.
    """
    if target_mag is None:
        # Flat-top window in frequency domain
        k = np.arange(N // 2 + 1)
        kf = k / (N // 2)
        # Flat-top: approximately constant in the middle, tapering at edges
        target_mag = np.ones(N // 2 + 1)
        # Taper the edges
        taper_width = 0.1
        for i in range(len(target_mag)):
            if kf[i] < taper_width:
                target_mag[i] = 1.0 - (1.0 - target_mag[i]) * (1 - kf[i] / taper_width)**2
            elif kf[i] > 1.0 - taper_width:
                target_mag[i] = 1.0 - (1.0 - target_mag[i]) * (1 - (1 - kf[i]) / taper_width)**2
    
    X = np.fft.rfft(x, n=_L_fft)
    current_mag = np.abs(X[:N//2+1])
    
    # Blend current magnitude with target
    new_mag = (1 - strength) * current_mag + strength * target_mag
    new_mag = np.maximum(new_mag, 1e-10)
    
    # Preserve phase, update magnitude
    phase = np.angle(X[:N//2+1])
    X_new = new_mag * np.exp(1j * phase)
    
    # Reconstruct full spectrum (Hermitian symmetry)
    X_full = np.zeros(_L_fft // 2 + 1, dtype=complex)
    X_full[:N//2+1] = X_new
    for i in range(1, N//2+1):
        if _L_fft - i <= _L_fft // 2:
            X_full[_L_fft - i] = np.conj(X_new[i])
    
    x_new = np.fft.irfft(X_full, n=_L_fft)[:N]
    return x_new

def _alternating_minimize(x0, max_outer=50, max_inner=100):
    """
    Alternating minimization:
    1. Update time-domain x to minimize peak autoconvolution
    2. Project to spectral magnitude constraint (flat-top)
    """
    x = x0.copy()
    best_x = x.copy()
    best_val = _objective(x)
    
    for outer in range(max_outer):
        if _check_deadline():
            break
        
        # Inner loop: optimize time-domain with fixed spectral target
        for inner in range(max_inner):
            if _check_deadline():
                break
            
            val, grad = _objective_grad_smooth(x, beta=50.0)
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-12:
                break
            
            # Gradient step with line search
            step_size = 0.01 / (gnorm + 1e-10)
            x_trial = x - step_size * grad
            
            val_trial = _objective(x_trial)
            if val_trial < val:
                x = x_trial
                val = val_trial
            else:
                # Try smaller step
                for _ in range(5):
                    step_size *= 0.5
                    x_trial = x - step_size * grad
                    val_trial = _objective(x_trial)
                    if val_trial < val:
                        x = x_trial
                        val = val_trial
                        break
        
        # Spectral projection
        x = _spectral_projection(x, strength=0.3)
        
        # Track best
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    rng_init = np.random.default_rng(123)
    for _ in range(30):
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate
    def _smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=30.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Alternating minimization with spectral projection
    x_alt, val_alt = _alternating_minimize(best_x, max_outer=30, max_inner=80)
    if val_alt < best_val:
        best_val = val_alt
        best_x = x_alt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Try multiple beta values for soft-max
    for beta in [15.0, 25.0, 50.0, 75.0, 100.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: More alternating minimization rounds with different strengths
    for strength in [0.2, 0.4, 0.6]:
        if _check_deadline():
            break
        
        x_alt2, val_alt2 = _alternating_minimize(best_x, max_outer=20, max_inner=60)
        if val_alt2 < best_val:
            best_val = val_alt2
            best_x = x_alt2.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Gradient descent polish
    best_x, best_val = _gradient_descent_polish(best_x, max_iters=3000, lr_init=0.01)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final L-BFGS-B polish with tight tolerances
    def _smooth_obj_tight(x):
        val, grad = _objective_grad_smooth(x, beta=80.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj_tight, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-15})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Gradient-guided random perturbation
    rng = np.random.default_rng(42)
    step_size = 0.02
    stall_count = 0
    max_iters = 20000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_grad_smooth(best_x, beta=50.0)
        gnorm = np.linalg.norm(grad)
        
        if gnorm > 1e-10:
            grad_dir = -grad / gnorm
        else:
            grad_dir = np.zeros(N)
        
        random_dir = rng.normal(0, 1, size=N)
        random_dir /= (np.linalg.norm(random_dir) + 1e-10)
        
        blend = 0.3
        direction = blend * grad_dir + (1 - blend) * random_dir
        direction /= (np.linalg.norm(direction) + 1e-10)
        
        x_trial = best_x + step_size * direction
        
        val_trial = _objective(x_trial)
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.999)
        else:
            stall_count += 1
            if stall_count % 2000 == 0:
                step_size = min(0.2, step_size * 1.1)
                if stall_count % 10000 == 0:
                    big_pert = rng.normal(0, step_size * 3, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
    
    # Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x

def _gradient_descent_polish(x, max_iters=2000, lr_init=0.01):
    """
    Efficient gradient descent with line search for polishing.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    lr = lr_init
    prev_val = best_val
    stall = 0
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_grad_smooth(best_x, beta=50.0)
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        step = lr * grad / gnorm
        x_trial = best_x - step
        val_trial = _objective(x_trial)
        
        alpha = 1.0
        c1 = 1e-4
        while val_trial > best_val - c1 * alpha * gnorm * np.dot(grad, step) and alpha > 1e-6:
            alpha *= 0.5
            x_trial = best_x - alpha * step
            val_trial = _objective(x_trial)
        
        if val_trial < best_val:
            best_x = x_trial.copy()
            best_val = val_trial
            stall = 0
            lr = min(lr * 1.02, 0.1)
        else:
            stall += 1
            lr *= 0.8
            if stall > 100:
                break
    
    return best_x, best_val
# EVOLVE-BLOCK-END
