import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_grad(h, beta):
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    grad = np.zeros(n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
    return val, grad

def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _h_from_spectral(coeffs):
    """Reconstruct h from spectral coefficients.
    coeffs = [a0, a1, b1, a2, b2, ..., aM, bM]
    h[i] = a0 + sum_{m=1}^{M} (a_m*cos(2*pi*m*i/N) + b_m*sin(2*pi*m*i/N))
    """
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    for m in range(1, len(coeffs) // 2 + 1):
        a_m = coeffs[2 * m - 1]
        b_m = coeffs[2 * m]
        h += a_m * np.cos(2 * np.pi * m * i / N) + b_m * np.sin(2 * np.pi * m * i / N)
    return h

def _spectral_score_and_grad(coeffs, beta):
    """Score and gradient in spectral parameterization."""
    h = _h_from_spectral(coeffs)
    h_clipped = np.clip(h, 0.0, 1.0)
    cs = _linear_overlap_values(h_clipped)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Gradient of cs with respect to h
    grad_h = np.zeros(N)
    n = N
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad_h[k:] += factor * h_clipped[:n-k]
        if k < n:
            grad_h[:n-k] += factor * h_clipped[k:]
    
    # Chain rule: d(coeffs) -> d(h) -> d(val)
    # h[i] = a0 + sum (a_m*cos + b_m*sin)
    # d(h[i])/d(a0) = 1
    # d(h[i])/d(a_m) = cos(2*pi*m*i/N)
    # d(h[i])/d(b_m) = sin(2*pi*m*i/N)
    # Also need to account for clipping: only contribute where 0 < h < 1
    mask = (h > 0) & (h < 1)
    grad_h_clipped = grad_h * mask
    
    grad_coeffs = np.zeros_like(coeffs)
    grad_coeffs[0] = np.sum(grad_h_clipped)
    i = np.arange(N)
    for m in range(1, len(coeffs) // 2 + 1):
        a_m_idx = 2 * m - 1
        b_m_idx = 2 * m
        cos_m = np.cos(2 * np.pi * m * i / N)
        sin_m = np.sin(2 * np.pi * m * i / N)
        grad_coeffs[a_m_idx] = np.sum(grad_h_clipped * cos_m)
        grad_coeffs[b_m_idx] = np.sum(grad_h_clipped * sin_m)
    
    return val, grad_coeffs

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Spectral global search
    # Use 5 frequency modes (10 coefficients + DC = 11 params)
    n_modes = 5
    n_coeffs = 1 + 2 * n_modes
    
    # Try multiple random spectral initializations
    for trial in range(30):
        if time.time() - t0 > _DEADLINE - 30:
            break
        
        # Random spectral coefficients
        a0 = 0.5  # DC component fixed to ensure mean is 0.5
        ac = np.random.uniform(-0.3, 0.3, n_modes)
        bc = np.random.uniform(-0.3, 0.3, n_modes)
        coeffs0 = np.concatenate([[a0], ac, bc])
        
        # Optimize in spectral space
        def obj_spec(x):
            val, _ = _spectral_score_and_grad(x, beta=50.0)
            return val
        
        def jac_spec(x):
            _, grad = _spectral_score_and_grad(x, beta=50.0)
            return grad
        
        # Bounds: a0 in [0,1], ac/bc in [-0.5, 0.5]
        bounds_spec = [(0.0, 1.0)] + [(-0.5, 0.5)] * (2 * n_modes)
        
        try:
            res = minimize(obj_spec, coeffs0, jac=jac_spec, method='L-BFGS-B',
                          bounds=bounds_spec, options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_score:
                h_candidate = _h_from_spectral(res.x)
                _update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 2: Random and structured starts (keep from original)
    for _ in range(20):
        if time.time() - t0 > _DEADLINE - 20:
            break
        h_rand = np.random.rand(N)
        _update_best(h_rand)
    
    for _ in range(20):
        if time.time() - t0 > _DEADLINE - 20:
            break
        i = np.arange(N)
        f = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2 * np.pi)
        h_sine = 0.5 + 0.5 * np.sin(2 * np.pi * f * i / N + phase)
        _update_best(h_sine)
    
    # Phase 3: Refine the best start with SLSQP
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val
            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad
                
            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_optimization(best_h, beta=50.0, maxiter=200, ftol=1e-9)
        
        # Polish
        if time.time() - t0 < _DEADLINE - 5:
            _run_optimization(best_h, beta=100.0, maxiter=300, ftol=1e-12)
    
    # Phase 4: Final spectral polish on best_h
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        # Extract spectral coefficients from best_h
        H = np.fft.rfft(best_h, N)
        # Reconstruct coefficients
        coeffs_final = np.zeros(1 + 2 * n_modes)
        coeffs_final[0] = H[0].real / N
        for m in range(1, n_modes + 1):
            coeffs_final[2*m-1] = 2 * H[m].real / N
            coeffs_final[2*m] = -2 * H[m].imag / N
        
        def obj_spec_final(x):
            val, _ = _spectral_score_and_grad(x, beta=100.0)
            return val
        def jac_spec_final(x):
            _, grad = _spectral_score_and_grad(x, beta=100.0)
            return grad
        
        bounds_spec = [(0.0, 1.0)] + [(-0.5, 0.5)] * (2 * n_modes)
        try:
            res = minimize(obj_spec_final, coeffs_final, jac=jac_spec_final, 
                          method='L-BFGS-B', bounds=bounds_spec, 
                          options={'maxiter': 100, 'ftol': 1e-12})
            if res.fun < best_score:
                h_candidate = _h_from_spectral(res.x)
                _update_best(h_candidate)
        except Exception:
            pass

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
