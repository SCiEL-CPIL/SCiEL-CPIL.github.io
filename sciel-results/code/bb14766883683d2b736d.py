"""Rewritten program: Subgradient/Smooth optimization for low off-diagonal autocorrelation."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    # Exclude k=0
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    # Soft max: temp * log(sum(exp(c/temp)))
    max_c = np.max(abs_c)
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _chirp_sequence(freqs, phases, amps, N):
    """Construct a sequence as sum of sinusoids (chirp-like)."""
    t = np.arange(N)
    x = np.zeros(N)
    for f, p, a in zip(freqs, phases, amps):
        x += a * np.sin(2 * np.pi * f * t / N + p)
    return x

def _multi_sine_objective(params, N=N, temp=0.5):
    """Objective for multi-sine parameterization.
    params = [freq1, phase1, amp1, freq2, phase2, amp2, ...]
    """
    n_sines = len(params) // 3
    if n_sines == 0:
        return 1e10
    freqs = params[0::3]
    phases = params[1::3]
    amps = params[2::3]
    x = _chirp_sequence(freqs, phases, amps, N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        return 1e10
    x = x / s
    return _objective_smooth(x, temp=temp)

def _optimize_multi_sine(n_sines, bounds, maxiter=200, deadline=None, temp=0.5):
    """Optimize multi-sine parameters using L-BFGS-B."""
    # Random initial point
    rng = np.random.default_rng(42)
    x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
    
    def obj(p):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _multi_sine_objective(p, N=N, temp=temp)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        best_params = res.x
        best_val = res.fun
    except Exception:
        best_params = x0
        best_val = obj(x0)
    
    # Try differential evolution for global search
    if deadline is not None and time.time() < deadline - 15:
        try:
            de_res = differential_evolution(
                obj, bounds, maxiter=30, popsize=10, tol=1e-6,
                seed=42, polish=False, update_interval=1
            )
            if de_res.fun < best_val:
                best_params = de_res.x
                best_val = de_res.fun
        except Exception:
            pass
    
    # Final polish with L-BFGS
    if deadline is not None and time.time() < deadline - 5:
        try:
            res2 = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res2.fun < best_val:
                best_params = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    # Convert to sequence
    n_sines_actual = len(best_params) // 3
    freqs = best_params[0::3]
    phases = best_params[1::3]
    amps = best_params[2::3]
    x = _chirp_sequence(freqs, phases, amps, N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        return None, 1e10
    x = x / s
    return x, _metric_offdiag(x)

def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, deadline=None):
    """Projected gradient descent with annealed temperature for final polish."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Normalize sum to 1
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        # Soft max gradient
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        # Compute soft max and its gradient
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        # Gradient of soft_max w.r.t. abs_c
        grad_abs_c = exp_vals / sum_exp
        
        # Gradient w.r.t. c (chain rule: d|c|/dc = sign(c))
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        # Gradient of autoconvolution w.r.t. x via cross-correlation
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        # Account for (sum x)^2 denominator
        s = np.sum(x)
        metric = soft_max / (s * s)
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        # Update
        x_new = x - lr * grad_metric
        
        # Project: keep sum non-zero
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        # Evaluate
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        # Decay learning rate and temperature
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-sine spectral design (low-dimensional global optimization)
    # Try different numbers of sinusoidal components
    for n_sines in [3, 5, 7, 10]:
        if time.time() > deadline - 40:
            break
        
        # Define bounds for each parameter
        bounds = []
        for _ in range(n_sines):
            bounds.append((1.0, 50.0))    # frequency
            bounds.append((0.0, 2 * np.pi))  # phase
            bounds.append((0.1, 1.0))     # amplitude
        
        x_candidate, val = _optimize_multi_sine(n_sines, bounds, maxiter=100, deadline=deadline - 20, temp=0.5)
        if x_candidate is not None and val < best_val:
            best_val = val
            best_x = x_candidate.copy()
    
    # Phase 2: Direct L-BFGS on full sequence from diverse starts
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Random binary with bias
    for _ in range(8):
        r = rng.random(N)
        x = np.where(r > 0.35, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
    
    # 3. Chirp-based starts
    for freq in [5, 10, 20]:
        t = np.arange(N)
        pattern = np.sin(2 * np.pi * freq * t / N)
        pattern = np.sign(pattern)
        if abs(np.sum(pattern)) < 1:
            pattern[0] = 1.0
        candidates.append(pattern)
    
    for x0 in candidates:
        if time.time() > deadline - 20:
            break
        
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            # Annealed L-BFGS
            res1 = minimize(
                lambda x: _objective_smooth(x, temp=1.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            res2 = minimize(
                lambda x: _objective_smooth(x, temp=0.1),
                res1.x,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            val = _metric_offdiag(res2.x)
            if val < best_val:
                best_val = val
                best_x = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3:
        try:
            polished = _gradient_descent_polish(
                best_x, 
                max_iter=800, 
                initial_lr=0.005,
                deadline=deadline - 1
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass

    return best_x
