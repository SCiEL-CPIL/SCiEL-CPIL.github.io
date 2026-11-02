import numpy as np
import time

N = 600

def _score(h):
    """Compute normalized max autoconvolution for nonnegative step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    denom = np.sum(h * h)
    if denom <= 0:
        return 1.0
    # Compute autocorrelation at all lags
    # autoconv[k] = sum_i h[i] * h[i+k] for k >= 0
    # We need max over all lags
    # Use FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, n=n_fft)[:N]  # lags 0..N-1
    max_auto = np.max(auto)
    return max_auto / denom

def _smooth_score(h, tau=0.01):
    """Smooth surrogate using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    denom = np.sum(h * h)
    if denom <= 0:
        return 1.0
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, n=n_fft)[:N]
    # log-sum-exp of auto / tau, times tau
    max_auto = np.max(auto)
    # For stability, subtract max
    shifted = auto - max_auto
    lse = max_auto + tau * np.log(np.sum(np.exp(shifted / tau)))
    return lse / denom

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    def elapsed_ok():
        return time.time() < deadline
    
    # Multiple initializations
    inits = []
    inits.append(np.ones(N))  # uniform
    x = np.linspace(-0.25, 0.25, N)
    inits.append(np.exp(-10 * x**2))  # Gaussian
    inits.append(0.5 + 0.5 * np.sin(2 * np.pi * x / 0.5))  # sine
    inits.append(np.linspace(0.1, 1.0, N))  # linear
    inits.append(np.ones(N) * (np.arange(N) < N // 2))  # step
    rng = np.random.RandomState(42)
    inits.append(np.abs(rng.randn(N)))  # random
    inits.append(np.exp(-5 * (x - 0.1)**2) + 0.3 * np.exp(-5 * (x + 0.1)**2))  # bimodal
    
    # Optimize using L-BFGS-B on log-parameters for nonnegativity
    from scipy.optimize import minimize
    
    def obj(params):
        h = np.exp(params)  # ensures positive
        return _smooth_score(h, tau=0.05)
    
    def grad(params):
        # Numerical gradient or analytical
        # Use finite differences for simplicity
        eps = 1e-7
        g = np.zeros_like(params)
        base = obj(params)
        for i in range(len(params)):
            params[i] += eps
            f_plus = obj(params)
            params[i] -= 2 * eps
            f_minus = obj(params)
            params[i] += eps
            g[i] = (f_plus - f_minus) / (2 * eps)
        return g
    
    for init in inits:
        if not elapsed_ok():
            break
        init = np.maximum(init, 0.01)
        params0 = np.log(init)
        
        try:
            res = minimize(obj, params0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.success or res.fun < best_score:
                h_opt = np.exp(res.x)
                s = _score(h_opt)
                if s < best_score:
                    best_score = s
                    best_h = h_opt
        except Exception:
            pass
    
    # Refine with smaller tau
    if elapsed_ok():
        for tau in [0.01, 0.005]:
            if not elapsed_ok():
                break
            def obj_refine(params):
                h = np.exp(params)
                return _smooth_score(h, tau=tau)
            
            def grad_refine(params):
                eps = 1e-7
                g = np.zeros_like(params)
                for i in range(len(params)):
                    params[i] += eps
                    f_plus = obj_refine(params)
                    params[i] -= 2 * eps
                    f_minus = obj_refine(params)
                    params[i] += eps
                    g[i] = (f_plus - f_minus) / (2 * eps)
                return g
            
            params0 = np.log(np.maximum(best_h, 0.01))
            try:
                res = minimize(obj_refine, params0, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_score:
                    h_opt = np.exp(res.x)
                    s = _score(h_opt)
                    if s < best_score:
                        best_score = s
                        best_h = h_opt
            except Exception:
                pass
    
    return list(best_h)
