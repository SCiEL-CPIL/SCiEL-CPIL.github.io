import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145


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


def _smooth_max_val_grad(h, beta):
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
        if k == 0:
            grad += -(4.0 / n) * wk * h
        else:
            grad += -(2.0 / n) * wk * (np.roll(h, -k) + np.roll(h, k))
    return val, grad


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


def _chirp_to_h(frac):
    """
    Generate step heights from a linear chirp with quadratic phase.
    h_j = (1 + frac * cos(2*pi*frac*j^2/N + phase)) / 2
    This ensures h in [0,1] and sum = N/2 (since cos averages to ~0).
    The chirp structure gives low autocorrelation, minimizing overlap.
    """
    j = np.arange(N)
    # Use a quadratic phase: phi_j = 2*pi * frac * j^2 / N
    # h_j = 0.5 + 0.5 * frac * cos(phi_j)
    # But we need sum = N/2, so the DC component should be 0.5
    # The cos term has ~zero mean for irrational frac, so sum ≈ N/2
    # We'll fix sum exactly with _fix_sum
    phi = 2.0 * np.pi * frac * j * j / N
    h = 0.5 + 0.5 * frac * np.cos(phi)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def _chirp_score(frac):
    h = _chirp_to_h(frac)
    return _score(h)


def _optimize_chirp_frac(frac0, max_iter=50):
    """Optimize the chirp sweep rate using gradient descent on the smooth max."""
    frac = frac0
    lr = 0.05
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        h = _chirp_to_h(frac)
        val, grad_h = _smooth_max_val_grad(h, beta=50.0)
        
        # Compute gradient of val w.r.t. frac via finite difference
        eps = 1e-6
        h_plus = _chirp_to_h(frac + eps)
        h_minus = _chirp_to_h(frac - eps)
        
        # Approximate gradient using the chain rule
        # d val / d frac ≈ (val(frac+eps) - val(frac-eps)) / (2*eps)
        val_plus = _smooth_max_val_grad(h_plus, beta=50.0)[0]
        val_minus = _smooth_max_val_grad(h_minus, beta=50.0)[0]
        grad_frac = (val_plus - val_minus) / (2.0 * eps)
        
        frac -= lr * grad_frac
        # Keep frac in a reasonable range
        frac = np.clip(frac, 0.01, 0.99)
        
        # Adaptive learning rate
        if it % 10 == 9:
            lr *= 0.95
    
    return frac, _chirp_score(frac)


def _spectral_to_h(coeffs):
    K = (len(coeffs) - 1) // 2
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    for k in range(1, K + 1):
        h += coeffs[k] * np.cos(2.0 * np.pi * k * i / N)
        h += coeffs[K + k] * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h):
    K = 5
    H = np.fft.rfft(h, n=N)
    coeffs = np.zeros(2 * K + 1)
    coeffs[0] = H[0].real / N
    for k in range(1, K + 1):
        coeffs[k] = 2.0 * H[k].real / N
        coeffs[K + k] = -2.0 * H[k].imag / N
    return coeffs


def _project_spectral(coeffs, K):
    h = _spectral_to_h(coeffs)
    coeffs = coeffs.copy()
    coeffs[0] = (N / 2.0 - np.sum(h - coeffs[0] * N)) / N
    h = _spectral_to_h(coeffs)
    h = np.clip(h, 0.0, 1.0)
    coeffs_new = _h_to_spectral(h)
    for _ in range(50):
        h = _spectral_to_h(coeffs_new)
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        coeffs_new[0] += diff / N
        h = _spectral_to_h(coeffs_new)
        h = np.clip(h, 0.0, 1.0)
        coeffs_new = _h_to_spectral(h)
    return coeffs_new


def _accelerated_spectral(coeffs0, beta, K, max_iter=300):
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    coeffs = _project_spectral(coeffs, K)
    h = _spectral_to_h(coeffs)
    h = _fix_sum(h)
    coeffs = _h_to_spectral(h)
    
    coeffs_prev = coeffs.copy()
    t_k = 1.0
    lr = 0.05
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(_spectral_to_h(coeffs), beta)
        
        # Compute gradient w.r.t. coeffs
        i = np.arange(N)
        grad_coeffs = np.zeros(2 * K + 1)
        grad_coeffs[0] = np.sum(grad)
        for k in range(1, K + 1):
            grad_coeffs[k] = np.sum(grad * np.cos(2.0 * np.pi * k * i / N))
            grad_coeffs[K + k] = np.sum(grad * np.sin(2.0 * np.pi * k * i / N))
        
        coeffs_new = coeffs - lr * grad_coeffs
        coeffs_new = _project_spectral(coeffs_new, K)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        coeffs_next = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs_prev)
        coeffs_next = _project_spectral(coeffs_next, K)
        
        coeffs_prev = coeffs
        coeffs = coeffs_next
        t_k = t_new
        
        if it % 50 == 49:
            lr = max(0.01, min(0.2, lr * 0.9))
    
    h_final = _spectral_to_h(coeffs)
    h_final = _fix_sum(h_final)
    return coeffs, h_final, _score(h_final)


def _accelerated_smooth_max(h0, beta, max_iter=500):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    
    h_prev = h.copy()
    t_k = 1.0
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(h, beta)
        
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
            lr = max(0.001, min(0.1, lr * 0.95))
    
    return h, _score(h)


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
    
    # Phase 1: Chirp-based initialization - search over sweep rates
    chirp_fracs = np.linspace(0.1, 0.9, 20)
    for frac in chirp_fracs:
        h = _chirp_to_h(frac)
        _update_best(h)
    
    # Phase 2: Optimize best chirp frac
    if best_h is not None and time.time() < _DEADLINE - 30:
        # Find which chirp frac gave best score
        best_frac = None
        best_frac_score = float('inf')
        for frac in chirp_fracs:
            s = _chirp_score(frac)
            if s < best_frac_score:
                best_frac_score = s
                best_frac = frac
        
        if best_frac is not None:
            frac_opt, score_opt = _optimize_chirp_frac(best_frac, max_iter=100)
            h_opt = _chirp_to_h(frac_opt)
            _update_best(h_opt)
    
    # Phase 3: Spectral optimization from best chirp solution
    if best_h is not None and time.time() < _DEADLINE - 20:
        K = 5
        best_coeffs = _h_to_spectral(best_h)
        for beta in [30.0, 60.0, 100.0]:
            if time.time() > _DEADLINE - 15:
                break
            c_candidate, h_candidate, s_candidate = _accelerated_spectral(
                best_coeffs, beta=beta, K=K, max_iter=200
            )
            if s_candidate < best_score:
                _update_best(h_candidate)
                best_coeffs = _h_to_spectral(h_candidate)
    
    # Phase 4: Full-space Nesterov refinement
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=50.0, max_iter=300)
        if s_candidate < best_score:
            _update_best(h_candidate)
        
        if time.time() < _DEADLINE - 10:
            h_candidate2, s_candidate2 = _accelerated_smooth_max(best_h, beta=100.0, max_iter=300)
            if s_candidate2 < best_score:
                _update_best(h_candidate2)
    
    # Phase 5: SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        def _run_optimization(h0, beta, maxiter, ftol):
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
        
        _run_optimization(best_h, beta=100.0, maxiter=150, ftol=1e-10)
        
        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=200.0, maxiter=200, ftol=1e-12)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
