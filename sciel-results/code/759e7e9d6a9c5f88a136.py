import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_FFT = 2 * N - 1

def _autoconv(f):
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)

def _score(f):
    ac = _autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _log_obj_and_grad(f, beta=20.0):
    # Objective: log(max|ac|) - 2 log|sum|
    # Approx max|ac| with soft-max
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return 0, np.zeros_like(f)
    
    # Soft max
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp) / beta + max_ac
    
    # Gradient of soft-max
    weights = exp_shifted / sum_exp
    signs = np.sign(ac)
    g_ac = weights * signs
    
    # Gradient of sum(ac * g_ac) w.r.t f
    # ac_k = sum_i f_i f_{i+k}
    # d/d f_j (sum_k g_ac_k ac_k) = sum_k g_ac_k (f_{k-j} + f_{j-k})
    # This is 2 * (f * g_ac)_j where * is correlation
    # Correlation of f and g_ac:
    # C_j = sum_k f_k g_ac_{k-j}
    # Using FFT: ifft(fft(f) * conj(fft(g_ac)))
    
    F_f = np.fft.rfft(f, n=N_FFT)
    F_g = np.fft.rfft(g_ac, n=N_FFT)
    corr = np.fft.irfft(F_f * np.conj(F_g), n=N_FFT)
    # corr[j] is sum_k f[k] g_ac[k-j] ?
    # Let's check.
    # np.correlate(f, g, mode='full')
    # FFT correlation: ifft(fft(f) * conj(fft(g))) gives circular correlation.
    # With zero padding to N_FFT, it gives linear correlation.
    # corr[j] = sum_k f[k] g_ac[k-j]
    # We want sum_k g_ac_k (f_{k-j} + f_{j-k})
    # Term 1: sum_k g_ac_k f_{k-j}. Let m = k-j => k = m+j. sum_m g_ac_{m+j} f_m.
    # This is (g_ac * f)_j ? No, correlation.
    # (f * g)_j = sum_k f_k g_{k-j}.
    # So Term 1 is (f * g_ac)_j.
    # Term 2: sum_k g_ac_k f_{j-k}. Let m = j-k => k = j-m. sum_m g_ac_{j-m} f_m.
    # This is (g_ac * f)_j.
    # Since g_ac is not symmetric, we need both.
    # But wait, ac is symmetric (real f). ac_k = ac_{-k}.
    # However, the weights g_ac_k depend on ac_k.
    # The gradient of the *exact* max is not necessarily symmetric in the weights if we pick the unique max.
    # But with soft-max, the weights are symmetric if ac is symmetric?
    # ac_k = sum_i f_i f_{i+k}.
    # ac_{-k} = sum_i f_i f_{i-k} = sum_j f_j f_{j+k} (let j=i-k) = ac_k.
    # So ac is even.
    # Thus abs_ac is even.
    # weights are even.
    # signs are even (sign(ac_k) = sign(ac_-k)).
    # So g_ac is even.
    # If g_ac is even, then f * g_ac = g_ac * f.
    # So the gradient is 2 * (f * g_ac).
    
    grad_smax = 2.0 * corr
    
    # Gradient of -2 log|S|
    s = np.sum(f)
    if abs(s) < 1e-12:
        grad_s = np.zeros_like(f)
    else:
        grad_s = -2.0 * np.ones(N) / s
        
    obj = np.log(log_sum) - 2.0 * np.log(abs(s))
    
    return obj, grad_smax / log_sum + grad_s

def _gradient_descent(f, max_iter=1000, lr=0.1, deadline=None):
    best_f = f.copy()
    best_val = _log_obj_and_grad(f)[0]
    
    for i in range(max_iter):
        if deadline and time.time() > deadline: break
        
        val, grad = _log_obj_and_grad(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
            
        # Adaptive step size
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-8: break
        
        # Simple backtracking line search
        step = lr
        for _ in range(10):
            f_new = f - step * grad
            val_new, _ = _log_obj_and_grad(f_new)
            if val_new < val:
                f = f_new
                break
            step *= 0.5
            
    return best_f

def _simulated_annealing(f, max_iter=2000, t0=1.0, t1=0.01, deadline=None):
    best_f = f.copy()
    best_val = _log_obj_and_grad(f)[0]
    curr_f = f.copy()
    curr_val = best_val
    
    for i in range(max_iter):
        if deadline and time.time() > deadline: break
        
        T = t0 * (t1/t0) ** (i/max_iter)
        
        # Perturb
        # Mix of small noise and large jumps
        if np.random.rand() < 0.1:
            # Large jump
            idx = np.random.randint(N)
            curr_f[idx] += np.random.randn() * 0.5
        else:
            # Small noise
            curr_f += np.random.randn(N) * 0.01
            
        val_new, _ = _log_obj_and_grad(curr_f)
        delta = val_new - curr_val
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            curr_f = curr_f
            curr_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_f = curr_f.copy()
                
    return best_f

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_f = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Initializations
    inits = []
    
    # 1. Random
    inits.append(np.random.randn(N))
    
    # 2. Chirp (Zadoff-Chu real part)
    t = np.arange(N)
    inits.append(np.cos(np.pi * t * (t+1) / N))
    
    # 3. Remez (Lowpass)
    try:
        from scipy.signal import remez
        h = remez(N-1, [0.0, 0.2], [1.0, 0.0])
        inits.append(h)
    except:
        pass
        
    # 4. Remez (Bandstop)
    try:
        h = remez(N-1, [0.0, 0.1, 0.2, 0.5], [1.0, 0.0, 1.0])
        inits.append(h)
    except:
        pass
        
    # 5. Barker-like (Optimized binary)
    # Start with random binary and do a few steps of local search
    b = np.random.choice([-1.0, 1.0], size=N)
    inits.append(b)
    
    # 6. All ones
    inits.append(np.ones(N))
    
    # Optimize each
    for f0 in inits:
        if time.time() > deadline - 10.0: break
        
        # Normalize to avoid numerical issues in log
        s = np.sum(f0)
        if s < 0: f0 = -f0
        if abs(s) < 1e-6: f0 += 1e-6
        
        # Phase 1: Simulated Annealing
        f_sa = _simulated_annealing(f0, max_iter=500, deadline=deadline - 20.0)
        
        # Phase 2: Gradient Descent
        f_gd = _gradient_descent(f_sa, max_iter=500, deadline=deadline - 10.0)
        
        # Phase 3: BFGS Polish
        try:
            res = minimize(
                lambda x: _log_obj_and_grad(x)[0],
                f_gd,
                jac=lambda x: _log_obj_and_grad(x)[1],
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            f_final = res.x
        except:
            f_final = f_gd
            
        sc = _score(f_final)
        if sc < best_score:
            best_score = sc
            best_f = f_final.copy()
            
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
