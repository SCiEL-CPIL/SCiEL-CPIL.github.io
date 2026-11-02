import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
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
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, tau=0.1):
    """
    Compute a smooth approximation of max(|ac|)/(sum x)^2 and its gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return m / (s * s), np.zeros_like(x)
    weights /= w_sum
    
    soft_max = m + tau * np.log(w_sum)
    val = soft_max / (s * s)
    
    # Gradient of soft_max
    v = np.sign(ac) * weights
    
    # grad_sm[k] = sum_i v[i] * (x[i-k] + x[k-i])
    # Using FFT for efficiency
    L_fft = 1
    L = 2 * N - 1
    while L_fft < L:
        L_fft *= 2
        
    # We need to compute sum_i v[i] x[i-k] and sum_i v[i] x[k-i]
    # Let's pad v and x
    v_pad = np.zeros(L_fft)
    v_pad[:L] = v
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    
    V = np.fft.rfft(v_pad)
    X = np.fft.rfft(x_pad)
    
    # Term 1: sum_i v[i] x[i-k]
    # This is correlation of v and x?
    # If y[k] = sum_i v[i] x[i-k], then Y = FFT(v) * conj(FFT(x)) ?
    # Let's check: y[k] = sum_i v[i] x[i-k].
    # In frequency domain, this is ifft(FFT(v) * conj(FFT(x)))?
    # Actually, correlation $R_{vx}[k] = \sum_i v[i+k] x[i]$.
    # Our term is $\sum_i v[i] x[i-k]$. Let $j = i-k \implies i = j+k$.
    # $\sum_j v[j+k] x[j]$. This is $R_{vx}[-k]$? No, $R_{vx}[k] = \sum_j v[j+k] x[j]$.
    # So Term 1 is $R_{vx}[-k]$?
    # Let's just use the property: ifft(FFT(v) * conj(FFT(x))) gives $\sum_i v[i] x[i-k]$?
    # Let's test with small arrays.
    # Actually, simpler:
    # grad = ifft( V * conj(X) ) + ifft( conj(V) * X ) ?
    # Let's derive:
    # d(ac_k)/dx_j = x_{k-j} + x_{j-k}
    # grad_j = sum_k v_k (x_{k-j} + x_{j-k})
    # Let A_j = sum_k v_k x_{k-j}
    # Let B_j = sum_k v_k x_{j-k}
    # A_j is cross-correlation of v and x at lag j?
    # Correlation $C_{vx}[j] = \sum_k v[k] x[k-j]$. Yes, A_j is $C_{vx}[j]$.
    # B_j = sum_k v_k x_{j-k}$. Let $m = j-k \implies k = j-m$.
    # B_j = sum_m v_{j-m} x_m$. This is convolution $(v * x)[j]$?
    # Convolution $(v * x)[j] = \sum_k v[k] x[j-k]$. Yes, B_j is $(v * x)[j]$.
    
    # So grad = C_{vx} + (v * x)
    # C_{vx} = ifft(FFT(v) * conj(FFT(x)))
    # (v * x) = ifft(FFT(v) * FFT(x))
    
    corr = np.fft.irfft(V * np.conj(X), n=L_fft)
    conv = np.fft.irfft(V * X, n=L_fft)
    
    grad_sm = (corr + conv)[:N]
    
    # Gradient of objective val = soft_max / s^2
    # d(val)/dx_j = (grad_sm_j * s^2 - soft_max * 2 * s) / s^4
    # = grad_sm_j / s^2 - 2 * soft_max / s^3
    
    grad_obj = grad_sm / (s * s) - 2 * soft_max / (s * s * s)
    
    return val, grad_obj

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    t = np.linspace(0, 1, N)
    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5]:
        initializations.append(r ** np.arange(N))
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    
    for x0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    # Phase 2: L-BFGS-B Optimization
    # We will run L-BFGS from the best initial point and a few random perturbations
    # to escape local minima.
    
    def fun(x):
        val, grad = _smooth_objective_and_grad(x, tau=0.1)
        if not np.isfinite(val):
            return 1e6, np.zeros_like(x)
        return val, grad

    # Run from best_x
    res = minimize(fun, best_x, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'gtol': 1e-6})
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_x = res.x.copy()
        
    # Run from a few random starts
    for _ in range(5):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        # Random start near best or uniform
        if rng.random() < 0.5:
            x0 = best_x + rng.normal(0, 0.1, size=N)
        else:
            x0 = rng.uniform(0.5, 1.5, size=N)
            
        # Normalize to avoid scale issues if needed, but objective is scale invariant?
        # val = max(|ac|) / s^2. If x -> c*x, ac -> c^2 ac, s -> c s.
        # val -> c^2 max(|ac|) / c^2 s^2 = val.
        # So scale doesn't matter.
        
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'gtol': 1e-6})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()

    # Phase 3: Fine tuning with smaller tau and local search
    # Reduce tau to sharpen the max approximation
    def fun_sharp(x):
        val, grad = _smooth_objective_and_grad(x, tau=0.01)
        if not np.isfinite(val):
            return 1e6, np.zeros_like(x)
        return val, grad

    if time.time() - START_TIME < TIME_LIMIT - 10:
        res = minimize(fun_sharp, best_x, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'gtol': 1e-8})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()

    # Final check
    final_val = _objective(best_x)
    if final_val < best_val:
        best_val = final_val

    return best_x
