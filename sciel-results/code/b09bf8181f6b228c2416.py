import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


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


def _spectral_to_h(coeffs):
    """Convert spectral coefficients to height sequence.
    coeffs[0] is the DC offset (should be 0.5 for sum=N/2).
    coeffs[1] = a1 (cos fundamental amplitude)
    coeffs[2] = b1 (sin fundamental amplitude)
    coeffs[3] = a2, coeffs[4] = b2, etc.
    """
    n = len(coeffs)
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    # Pairs: (a_k, b_k) for k=1,2,...
    num_modes = (n - 1) // 2
    for k in range(1, num_modes + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        h += a * np.cos(2.0 * np.pi * k * i / N) + b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h):
    """Convert height sequence to spectral coefficients via DFT."""
    H = np.fft.rfft(h)
    coeffs = [H[0].real / N]  # DC component
    for k in range(1, len(H)):
        coeffs.append(2.0 * H[k].real / N)
        coeffs.append(-2.0 * H[k].imag / N)
    return np.array(coeffs)


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_val_grad(h, beta):
    """Compute smooth max value and gradient w.r.t. h."""
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
        wk = w[k]
        if wk < 1e-15:
            continue
        factor = -(2.0 / n) * wk
        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]
    return val, grad


def _spectral_optimize(num_modes, h0, beta, maxiter=500, ftol=1e-12):
    """Optimize in spectral space."""
    coeffs0 = _h_to_spectral(h0)
    # Truncate to num_modes + 1 coefficients (DC + num_modes pairs)
    n_coeff = 2 * num_modes + 1
    coeffs0 = coeffs0[:n_coeff]
    
    bounds = [(0.5 - 0.5, 0.5 + 0.5)]  # DC component free
    for _ in range(num_modes):
        bounds.append((-0.5, 0.5))
        bounds.append((-0.5, 0.5))
    
    def obj(c):
        h = _spectral_to_h(c)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        val, _ = _smooth_max_val_grad(h, beta)
        return val
    
    try:
        res = minimize(obj, coeffs0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': ftol, 'adaptive': True})
        h_opt = _spectral_to_h(res.x)
        h_opt = np.clip(h_opt, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)


def _projected_gradient_spectral(coeffs0, beta, max_iter=300, lr=0.01):
    """Projected gradient in spectral space with box+sum constraints on h."""
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    n_coeff = len(coeffs)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        h = _spectral_to_h(coeffs)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        val, grad_h = _smooth_max_val_grad(h, beta)
        
        # Compute gradient w.r.t. spectral coefficients
        i = np.arange(N)
        grad_c = np.zeros(n_coeff)
        # DC component
        grad_c[0] = np.sum(grad_h) / N
        for k in range(1, (n_coeff - 1) // 2 + 1):
            a_idx = 2 * k - 1
            b_idx = 2 * k
            cos_k = np.cos(2.0 * np.pi * k * i / N)
            sin_k = np.sin(2.0 * np.pi * k * i / N)
            grad_c[a_idx] = (2.0 / N) * np.sum(grad_h * cos_k)
            grad_c[b_idx] = (2.0 / N) * np.sum(grad_h * sin_k)
        
        coeffs_new = coeffs - lr * grad_c
        # Project: ensure resulting h is in [0,1] and sum = N/2
        h_new = _spectral_to_h(coeffs_new)
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)
        coeffs = _h_to_spectral(h_new)[:n_coeff]
        
        if it % 50 == 49:
            lr = max(0.001, lr * 0.95)
    
    h_final = _spectral_to_h(coeffs)
    h_final = np.clip(h_final, 0.0, 1.0)
    h_final = _fix_sum(h_final)
    return h_final, _score(h_final)


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
    
    i = np.arange(N)
    
    # Phase 1: Generate diverse structured starts
    starts = []
    
    # Random starts
    for _ in range(15):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))
    
    # Sine waves with various frequencies
    for f in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))
    
    # Cosine waves
    for f in [1, 2, 3]:
        for amp in [0.2, 0.3, 0.4]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))
    
    # Two-frequency combinations
    h_2f = 0.5 + 0.3 * np.cos(2 * np.pi * i / N) + 0.2 * np.cos(4 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))
    h_2f2 = 0.5 + 0.25 * np.sin(2 * np.pi * i / N) + 0.25 * np.sin(3 * np.pi * i / N)
    starts.append(_fix_sum(h_2f2))
    
    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)
    
    # Phase 2: Spectral optimization from best starts
    if best_h is not None:
        # Try different numbers of modes
        for num_modes in [2, 3, 4, 5]:
            if time.time() > _DEADLINE - 10:
                break
            h_candidate, s_candidate = _spectral_optimize(num_modes, best_h, beta=50.0, maxiter=300)
            if s_candidate < best_score:
                _update_best(h_candidate)
        
        # Gradient-based spectral refinement
        if time.time() < _DEADLINE - 10:
            coeffs0 = _h_to_spectral(best_h)[:11]  # 5 modes
            h_candidate, s_candidate = _projected_gradient_spectral(coeffs0, beta=80.0, max_iter=400)
            if s_candidate < best_score:
                _update_best(h_candidate)
    
    # Phase 3: Full-space Nesterov accelerated gradient
    if best_h is not None and time.time() < _DEADLINE - 15:
        h = np.array(best_h, dtype=np.float64, copy=True)
        h = _fix_sum(h)
        h_prev = h.copy()
        t_k = 1.0
        lr = 0.01
        
        for it in range(500):
            if time.time() > _DEADLINE - 5:
                break
            val, grad = _smooth_max_val_grad(h, beta=100.0)
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new = _fix_sum(h_new)
            
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
            h_next = np.clip(h_next, 0.0, 1.0)
            h_next = _fix_sum(h_next)
            
            h_prev = h
            h = h_next
            t_k = t_new
            
            if it % 50 == 49:
                lr = max(0.001, min(0.05, lr * 0.97))
        
        _update_best(h)
    
    # Phase 4: SLSQP polish with high beta
    if best_h is not None and time.time() < _DEADLINE - 5:
        def _run_slsqp(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            def obj_func(x):
                val, _ = _smooth_max_val_grad(x, beta)
                return val
            
            def jac_func(x):
                _, grad = _smooth_max_val_grad(x, beta)
                return grad
            
            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass
        
        _run_slsqp(best_h, beta=150.0, maxiter=200, ftol=1e-11)
        
        if time.time() < _DEADLINE - 3:
            _run_slsqp(best_h, beta=250.0, maxiter=300, ftol=1e-13)
    
    # Phase 5: Final spectral polish
    if best_h is not None and time.time() < _DEADLINE - 3:
        for num_modes in [3, 4]:
            if time.time() > _DEADLINE - 2:
                break
            h_candidate, s_candidate = _spectral_optimize(num_modes, best_h, beta=200.0, maxiter=200)
            if s_candidate < best_score:
                _update_best(h_candidate)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
