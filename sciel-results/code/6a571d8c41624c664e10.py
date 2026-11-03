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
    
    # Gradient:
    # val = soft_max / s^2
    # d(val)/dx[j] = (1/s^2) * d(soft_max)/dx[j] - (2*soft_max/s^3) * 1
    
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx[j]
    # d(ac[n])/dx[j] = x[j+n] + x[j-n] (with boundary handling)
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # Compute d(soft_max)/dx efficiently using FFT
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n] (correlation)
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n] 
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
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
    
    # Constraint violation: soft_max - t * s^2
    # Objective: t + penalty for constraint violation
    violation = soft_max - t * s * s
    penalty = 100.0 * max(0, violation) ** 2
    
    obj_val = t + penalty
    
    # Gradient wrt t
    grad_t = 1.0
    if violation > 0:
        grad_t += 200.0 * violation
    
    # Gradient wrt x
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    # d(violation)/dx = d_softmax_dx - t * 2 * s
    d_violation_dx = d_softmax_dx - 2.0 * t * s
    
    grad_x = np.zeros(N)
    if violation > 0:
        grad_x = 200.0 * violation * d_violation_dx
    
    grad = np.zeros_like(params)
    grad[:N] = grad_x
    grad[N] = grad_t
    
    return obj_val, grad

def _coordinate_descent(x, max_iters=5000, step_scale=0.1):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _subgradient_polish(x, max_iters=20000, step_scale=0.01):
    """
    Subgradient-like polish using exact objective.
    At each step, identify the lag with max |ac| and adjust x to reduce it.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        max_idx = np.argmax(abs_ac)
        sign_ac = np.sign(ac[max_idx]) if ac[max_idx] != 0 else 1.0
        
        # To reduce |ac[max_idx]|, we want to decrease ac[max_idx]*sign_ac
        # ac[n] = sum_j x[j]*x[j+n] (with boundary handling)
        # d(ac[n])/dx[j] = x[j+n] + x[j-n]
        
        # Compute gradient of ac[n] wrt x
        grad_ac = np.zeros(N)
        n = max_idx
        for j in range(N):
            g = 0.0
            if j + n < N:
                g += best_x[j + n]
            if j - n >= 0:
                g += best_x[j - n]
            grad_ac[j] = g
        
        # Step: decrease ac[n]*sign_ac by moving x in direction -sign_ac * grad_ac
        step = step_scale / (np.linalg.norm(grad_ac) + 1e-10)
        x_trial = best_x - sign_ac * step * grad_ac
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_scale = max(0.0001, step_scale * 0.999)
        else:
            step_scale = min(0.1, step_scale * 1.02)
    
    return best_x, best_val

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    a: shape parameter for sinc-like spectrum
    tilt: frequency tilt
    mod: modulation amount
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
    for _ in range(20):
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
                        options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Try multiple beta values for soft-max
    for beta in [10.0, 20.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Epigraph optimization
    for beta in [30.0, 50.0, 80.0]:
        if _check_deadline():
            break
        
        # Initialize t with current best value
        t_init = best_val
        params0 = np.concatenate([best_x, [t_init]])
        
        try:
            result = minimize(_epigraph_objective, params0, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_x = result.x[:N].copy()
                best_val = _objective(best_x)
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
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_param_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_design(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Coordinate descent polish (reduced iters to save time)
    best_x, best_val = _coordinate_descent(best_x, max_iters=1000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: L-BFGS-B polish with medium beta
    def _medium_smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=100.0)
        return val, grad
    
    try:
        result = minimize(_medium_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
        final_val = _objective(result.x)
        if final_val < best_val:
            best_val = final_val
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Subgradient polish with exact objective
    if not _check_deadline():
        best_x, best_val = _subgradient_polish(best_x, max_iters=15000, step_scale=0.01)
    
    if _check_deadline():
        return best_x
    
    # Phase 9: Final tight soft-max L-BFGS-B with high beta
    def _tight_smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=200.0)
        return val, grad
    
    try:
        result = minimize(_tight_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-15})
        final_val = _objective(result.x)
        if final_val < best_val:
            best_val = final_val
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 10: Final subgradient polish
    if not _check_deadline():
        best_x, best_val = _subgradient_polish(best_x, max_iters=10000, step_scale=0.005)
    
    # Final exact evaluation
    best_val = _objective(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
