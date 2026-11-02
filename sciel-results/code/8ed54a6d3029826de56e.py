"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

# Precompute FFT size
_L_fft = 1
while _L_fft < 2 * N - 1:
    _L_fft *= 2

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
    Uses log-sum-exp for numerical stability.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # Use a larger FFT for the gradient computation
    L_fft_grad = 1
    while L_fft_grad < 3 * N:
        L_fft_grad *= 2
    
    G = np.fft.rfft(g, n=L_fft_grad)
    Xf = np.fft.rfft(x, n=L_fft_grad)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft_grad)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft_grad)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation: minimize t subject to soft_max(|ac|) <= t * s^2.
    params = [x_0, ..., x_{N-1}, t]
    Returns (objective_value, gradient) for L-BFGS-B.
    """
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return t, np.zeros_like(params)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    # Objective: t + penalty * max(0, soft_max - t * s^2)^2
    violation = soft_max - t * s * s
    penalty_coeff = 1000.0
    penalty = penalty_coeff * max(0.0, violation) ** 2
    obj = t + penalty
    
    # Gradient w.r.t. t
    d_obj_dt = 1.0 - 2.0 * penalty_coeff * max(0.0, violation) * s * s
    
    # Gradient w.r.t. x
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft_grad = 1
    while L_fft_grad < 3 * N:
        L_fft_grad *= 2
    
    G = np.fft.rfft(g, n=L_fft_grad)
    Xf = np.fft.rfft(x, n=L_fft_grad)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft_grad)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft_grad)[:N]
    d_softmax_dx = term1 + term2
    
    if violation > 0:
        d_penalty_dx = 2.0 * penalty_coeff * violation * (d_softmax_dx - 2.0 * t * s * np.ones(N))
    else:
        d_penalty_dx = np.zeros(N)
    
    grad_x = d_penalty_dx
    grad_t = d_obj_dt
    
    grad = np.concatenate([grad_x, [grad_t]])
    
    return obj, grad

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
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        # Backtracking line search
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

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    """
    a = np.clip(a, 0.5, 3.0)
    tilt = np.clip(tilt, -1.0, 1.0)
    mod = np.clip(mod, 0.0, 1.0)
    
    k = np.arange(N)
    c = 0.5 + a
    kf = k / N
    
    mag = np.exp(-c * kf**2)
    mag = mag * np.exp(tilt * kf)
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x

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
    
    # Add some random smooth initializations
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
    
    # Phase 3: Try multiple beta values for soft-max
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
    
    # Phase 4: Epigraph formulation
    try:
        t0 = best_val
        params0 = np.concatenate([best_x, [t0]])
        
        def _epi_obj(params):
            return _epigraph_objective(params, beta=50.0)
        
        result = minimize(_epi_obj, params0, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            x_candidate = result.x[:N]
            val_candidate = _objective(x_candidate)
            if val_candidate < best_val:
                best_val = val_candidate
                best_x = x_candidate.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Spectral parameter optimization
    def _param_objective(params):
        x = _spectral_design(params[0], params[1], params[2])
        return _objective(x)
    
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, -0.5, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.5, 0.0, 0.3],
        [2.5, 0.5, 0.2],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_param_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_design(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Gradient descent polish (replaces expensive coordinate descent)
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
        
        # Mix gradient direction with random perturbation
        val, grad = _objective_grad_smooth(best_x, beta=50.0)
        gnorm = np.linalg.norm(grad)
        
        if gnorm > 1e-10:
            grad_dir = -grad / gnorm
        else:
            grad_dir = np.zeros(N)
        
        # Random component
        random_dir = rng.normal(0, 1, size=N)
        random_dir /= (np.linalg.norm(random_dir) + 1e-10)
        
        # Blend
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
                    # Big kick
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
# EVOLVE-BLOCK-END
