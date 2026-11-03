# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog
from scipy.fftpack import dct, idct

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

def _dct_objective(c, M):
    """
    Objective for DCT coefficient optimization.
    c: vector of DCT coefficients (length M)
    M: number of coefficients to use (rest are zero)
    """
    # Reconstruct x from DCT coefficients
    # idct expects type 2 by default? 
    # scipy.fftpack.idct(x, type=2)
    # We need to handle scaling carefully or just use the output directly.
    # The scaling of DCT/IDCT affects the value of sum(x) and autoconv.
    # However, the ratio max|ac| / (sum x)^2 is scale invariant.
    # So we can just use the raw output of idct.
    
    c_full = np.zeros(N)
    c_full[:M] = c
    x = idct(c_full, type=2)
    
    return _objective(x)

def _dct_objective_grad(c, M, beta=50.0):
    """
    Gradient for DCT coefficient optimization.
    We compute the gradient of the smooth surrogate w.r.t x,
    then project it back to the DCT coefficient space.
    grad_c = DCT(grad_x) (with appropriate scaling)
    """
    c_full = np.zeros(N)
    c_full[:M] = c
    x = idct(c_full, type=2)
    
    val, grad_x = _objective_grad_smooth(x, beta=beta)
    
    # Project grad_x to DCT space
    # grad_c = dct(grad_x, type=2)
    # We only need the first M components
    grad_c_full = dct(grad_x, type=2)
    grad_c = grad_c_full[:M]
    
    return val, grad_c

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

    # Phase 2: L-BFGS-B with smooth surrogate (Direct x)
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

    # Phase 3: DCT Manifold Optimization (Structural Change)
    # This optimizes the first M DCT coefficients, enforcing smoothness.
    for M in [10, 20, 50]:
        if _check_deadline():
            break
        
        # Initialize c from current best_x
        c0 = dct(best_x, type=2)[:M]
        
        def _dct_obj_grad(c, M_val=M):
            val, grad = _dct_objective_grad(c, M_val, beta=50.0)
            return val, grad
        
        try:
            result = minimize(_dct_obj_grad, c0, method='L-BFGS-B', jac=True,
                            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
            
            # Reconstruct x from optimized c
            c_opt = result.x
            c_full = np.zeros(N)
            c_full[:M] = c_opt
            x_dct = idct(c_full, type=2)
            
            val_dct = _objective(x_dct)
            if val_dct < best_val:
                best_val = val_dct
                best_x = x_dct.copy()
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Phase 4: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=1000, step_scale=0.05)

    if _check_deadline():
        return best_x

    # Phase 5: Final L-BFGS-B polish
    def _final_smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=100.0)
        return val, grad
    
    try:
        result = minimize(_final_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
        final_val = _objective(result.x)
        if final_val < best_val:
            best_val = final_val
            best_x = result.x.copy()
    except Exception:
        pass

    # Final exact evaluation
    best_val = _objective(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
