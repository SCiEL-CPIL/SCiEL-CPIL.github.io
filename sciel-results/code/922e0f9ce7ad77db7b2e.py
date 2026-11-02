import numpy as np
import time
from scipy.optimize import minimize

N = 400

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
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_ac(f, beta):
    """Compute soft max of |autoconv| and its weights."""
    ac = _autoconv_fft(f)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(abs_ac)
    exp_terms = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_max_val = m + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    return soft_max_val, weights

def _soft_score_grad(f, beta):
    """
    Compute soft score and its gradient w.r.t. f.
    Uses FFT-based convolution for efficiency.
    """
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.zeros_like(f), np.inf
    
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    soft_max_val, weights = _soft_max_abs_ac(f, beta)
    
    if soft_max_val < 1e-15:
        return np.zeros_like(f), 0.0
    
    w = weights  # length 2*n-1
    w_padded = np.zeros(size)
    w_padded[:len(w)] = w
    
    W = np.fft.rfft(w_padded, size)
    F_full = np.fft.rfft(f, size)
    
    # conv1: sum_i w[i] * f[i-k] (cross-correlation style)
    # In FFT domain: W * conj(F) gives correlation
    # But we want convolution: w * f
    # Let's stick to the standard convolution gradient derivation:
    # ac[k] = sum_i f[i] f[k-i]
    # d(ac[k])/df[j] = f[k-j] + f[j-k] (with appropriate bounds)
    # grad = sum_k w[k] * sign(ac[k]) * (f[k-j] + f[j-k])
    # This is equivalent to conv(w_sign, f) + conv(w_sign_flip, f)
    
    # We need to handle the sign of ac[k]
    ac = _autoconv_fft(f)
    signs = np.sign(ac)
    w_signed = weights * signs
    
    # Term 1: conv(w_signed, f)
    ws_padded = np.zeros(size)
    ws_padded[:len(w_signed)] = w_signed
    WS = np.fft.rfft(ws_padded, size)
    conv1 = np.fft.irfft(WS * F_full, size)[:2*n-1]
    
    # Term 2: conv(w_signed_flip, f)
    # w_signed_flip[i] = w_signed[len(w)-1-i]
    w_flip = w_signed[::-1]
    wf_padded = np.zeros(size)
    wf_padded[:len(w_flip)] = w_flip
    WF = np.fft.rfft(wf_padded, size)
    conv2 = np.fft.irfft(WF * F_full, size)[:2*n-1]
    
    grad_soft_max = conv1 + conv2
    
    # Gradient of objective = soft_max / s^2
    grad = grad_soft_max / (s * s) - 2.0 * soft_max_val / (s * s * s)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(f)
    
    return grad, soft_max_val / (s * s)

def _structured_init(freq, phase=0.0):
    t = np.arange(N)
    x = np.cos(2 * np.pi * freq * t / N + phase)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def _random_init(rng):
    x = rng.normal(1.0, 0.5, size=N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def _peak_targeted_polish(x, max_iters=2000, time_limit=10.0, rng=None):
    """
    Targeted polish: identify dominant autoconvolution lags and apply
    targeted perturbations to reduce them.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    start = time.time()
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        L = len(ac)
        
        # Find the top few dominant lags
        top_k = min(10, L)
        top_indices = np.argsort(abs_ac)[-top_k:]
        
        improved = False
        for k in top_indices:
            if time.time() - start > time_limit:
                break
            
            valid_i = np.arange(max(0, k - N + 1), min(k + 1, N))
            if len(valid_i) == 0:
                continue
            
            current_ac_k = ac[k]
            best_reduction = 0
            best_i = None
            
            for i in valid_i:
                j = k - i
                if j < 0 or j >= N:
                    continue
                contrib = best_x[i] * best_x[j]
                if i == j:
                    continue
                new_ac_k = current_ac_k - 2 * contrib
                if abs(new_ac_k) < abs(current_ac_k) - best_reduction:
                    best_reduction = abs(current_ac_k) - abs(new_ac_k)
                    best_i = i
            
            if best_i is not None and best_reduction > 0:
                trial_x = best_x.copy()
                trial_x[best_i] = -trial_x[best_i]
                trial_val = _objective(trial_x)
                if trial_val < best_val:
                    best_x = trial_x
                    best_val = trial_val
                    improved = True
        
        if not improved:
            perturbation = np.zeros(N)
            for k in top_indices[:5]:
                valid_i = np.arange(max(0, k - N + 1), min(k + 1, N))
                if len(valid_i) > 0:
                    idx = valid_i[rng.integers(len(valid_i))]
                    perturbation[idx] += rng.normal(0, 0.1)
            
            if np.linalg.norm(perturbation) > 0:
                trial_x = best_x + perturbation
                trial_val = _objective(trial_x)
                if trial_val < best_val:
                    best_x = trial_x
                    best_val = trial_val
                    improved = True
        
        if not improved and it > 100:
            break
    
    return best_x, best_val

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145
    rng = np.random.default_rng(42)
    
    def _time_ok():
        return (time.time() - START_TIME) < TIME_LIMIT
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    # Phase 1: Generate diverse initializations
    initializations = [np.ones(N)]
    
    # Structured inits
    for freq in [1, 2, 3, 5, 7, 10, 13, 17]:
        initializations.append(_structured_init(freq))
        initializations.append(_structured_init(freq, phase=np.pi/2))
        
    # Random inits
    for _ in range(10):
        initializations.append(_random_init(rng))
        
    # Power law inits
    for r in [0.9, 0.8, 0.7, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
        
    # Evaluate and pick best initial
    for x0 in initializations:
        if not _time_ok():
            break
        if np.sum(x0) < 0:
            x0 = -x0
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
            
    # Phase 2: L-BFGS-B Optimization with Soft Max
    # Use multiple betas and restarts
    betas = [50.0, 100.0, 200.0]
    
    # Take top 3 initializers + best found so far
    # Actually, let's just run L-BFGS from the best initial found and a few others
    candidates = [best_x.copy()]
    for x0 in initializations[:5]: # Just a few more
        candidates.append(x0.copy())
        
    for x0 in candidates:
        if not _time_ok():
            break
            
        for beta in betas:
            if not _time_ok():
                break
                
            def objective_and_grad(x_flat):
                g, val = _soft_score_grad(x_flat, beta)
                if not np.all(np.isfinite(g)):
                    g = np.zeros_like(x_flat)
                if not np.isfinite(val):
                    val = 1e10
                return val, g
            
            try:
                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': 500
                    }
                )
                
                x_opt = result.x
                sc = _objective(x_opt)
                
                if sc < best_val:
                    best_val = sc
                    best_x = x_opt.copy()
                    
            except Exception:
                pass
    
    # Phase 3: Peak-targeted polish
    if _time_ok():
        remaining_time = TIME_LIMIT - (time.time() - START_TIME) - 5.0
        if remaining_time > 2.0:
            polished_x, polished_val = _peak_targeted_polish(
                best_x, 
                max_iters=5000, 
                time_limit=min(10.0, remaining_time), 
                rng=rng
            )
            if polished_val < best_val:
                best_val = polished_val
                best_x = polished_x
    
    # Phase 4: Fine-tuning with small perturbations (Steepest Descent / Random Walk)
    if _time_ok():
        step_size = 0.05
        stall_count = 0
        # Use gradient if available, otherwise random
        for iteration in range(100000):
            if not _time_ok():
                break
            
            # Try gradient step
            grad, _ = _soft_score_grad(best_x, beta=100.0)
            norm_grad = np.linalg.norm(grad)
            
            if norm_grad > 1e-8:
                direction = -grad / norm_grad
            else:
                direction = rng.normal(0, 1, size=N)
                direction /= np.linalg.norm(direction)
                
            # Adaptive step size
            x_trial = best_x + step_size * direction
            val = _objective(x_trial)
            
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                stall_count = 0
                step_size = min(0.5, step_size * 1.05)
            else:
                stall_count += 1
                if stall_count % 100 == 0:
                    step_size = max(0.001, step_size * 0.8)
                    
    return best_x
