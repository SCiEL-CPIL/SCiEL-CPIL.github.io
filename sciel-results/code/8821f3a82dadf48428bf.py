"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses epigraph formulation with SLSQP, refined with higher precision polishes.
Combines cosine initialization with random restarts and adaptive beta tuning."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _autocorr(h):
    """
    Compute linear autocorrelation r_k = sum_{i=k}^{N-1} h_i * h_{i-k} for k=0..N-1.
    Uses FFT for O(N log N) computation.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


def _overlap_values(h):
    """
    Compute overlap values C_k = (2/N) * (N/2 - r_k) for k=0..N-1.
    """
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_value(h, beta):
    """
    Smooth maximum of overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    m = np.max(cs)
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _project_sum_box(h):
    """
    Project h onto the set {x : sum(x) = N/2, 0 <= x_i <= 1}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # Bisection on lambda: find lambda such that sum(clip(h - lambda, 0, 1)) = target
    lo, hi = -10.0, 10.0
    
    for _ in range(100):
        mid = (lo + hi) / 2.0
        projected = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(projected)
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lambda_star = (lo + hi) / 2.0
    h_proj = np.clip(h - lambda_star, 0.0, 1.0)
    
    # Tiny correction for floating point
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        # Distribute tiny error to elements not at bounds
        mask = (h_proj > 1e-12) & (h_proj < 1.0 - 1e-12)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
    
    return h_proj


def _pgd_optimize(h0, beta, n_steps=5000, lr_init=0.01):
    """
    Projected gradient descent with adaptive step size.
    """
    h = h0.copy()
    lr = lr_init
    best_h = h.copy()
    best_val = _smooth_max_value(h, beta)
    
    for step in range(n_steps):
        if time.time() > _DEADLINE - 5:
            break
            
        val, grad = _smooth_max_grad(h, beta)
        
        # Adaptive step size (barzilai-borwein style)
        h_new = _project_sum_box(h - lr * grad)
        val_new = _smooth_max_value(h_new, beta)
        
        if val_new < val:
            h = h_new
            val = val_new
            # Increase step size slightly
            lr = min(lr * 1.05, 0.05)
        else:
            # Decrease step size
            lr = lr * 0.7
            
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # Second pass if clipping changed sum
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """
    Run SLSQP optimization on the smooth max objective.
    """
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
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _polish(h, beta, maxiter, ftol, gtol):
    """
    High-precision polish pass with tighter tolerances.
    """
    h0 = h.copy()
    h_new = _run_slsqp(h0, beta, maxiter, ftol, gtol)
    h_new = _fix_sum(h_new)
    
    # Verify improvement
    if _score(h_new) < _score(h0):
        return h_new
    return h0


def _spectral_h_from_coeffs(coeffs):
    """
    Reconstruct h from real DFT coefficients (modes 1..K).
    coeffs = [a1, b1, a2, b2, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2*(k-1)]
        b = coeffs[2*(k-1)+1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """
    Extract the first K real DFT coefficients from h.
    Returns [a1, b1, ..., aK, bK] where h ~ 0.5 + sum (a_k cos + b_k sin)
    """
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        # For real FFT: h_i = H[0]/N + sum_{k=1}^{N/2-1} (2/N)*Re(H[k])*cos(2*pi*k*i/N) - (2/N)*Im(H[k])*sin(2*pi*k*i/N) + ...
        # So a_k = (2/N)*Re(H[k]), b_k = -(2/N)*Im(H[k])
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter):
    """
    Optimize the spectral coefficients of h to minimize the smooth max objective.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)
    
    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        
        # Penalty for out-of-bounds to keep solution valid
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2)
        
        val, _ = _smooth_max_grad(h, beta)
        return val + 1.0 * penalty

    try:
        res = minimize(obj, coeffs0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success:
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass
    return h_init


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
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
    
    # 1. Multiple diverse initializations
    i = np.arange(N)
    
    # Cosine start
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Sine start
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # Alternating pattern
    h_alt = np.where(i % 2 == 0, 0.75, 0.25)
    h_alt = _fix_sum(h_alt)
    _update_best(h_alt)
    
    # Block pattern
    h_block = np.where(i < N//2, 0.75, 0.25)
    h_block = _fix_sum(h_block)
    _update_best(h_block)
    
    # Random initializations
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _fix_sum(h_rand)
        _update_best(h_rand)
    
    # 2. Spectral Optimization on best candidate
    for K in [2, 4, 8, 16, 32]:
        if time.time() - t0 > _DEADLINE - 30:
            break
        h_spec = _optimize_spectral(best_h, K, beta=50.0, maxiter=200)
        _update_best(h_spec)
        
    # 3. Projected Gradient Descent (fast, many steps)
    if time.time() - t0 < _DEADLINE - 40:
        h_pgd = _pgd_optimize(best_h, beta=50.0, n_steps=10000, lr_init=0.01)
        _update_best(h_pgd)
        
    # 4. SLSQP on best candidate
    if time.time() - t0 < _DEADLINE - 30:
        h_opt = _run_slsqp(best_h, beta=50.0, maxiter=300, ftol=1e-10, gtol=1e-8)
        _update_best(h_opt)
        
    # 5. Random restarts with PGD + Spectral Polish
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            h_pgd_rand = _pgd_optimize(h_rand, beta=50.0, n_steps=3000, lr_init=0.01)
            _update_best(h_pgd_rand)
            
    # 6. Polish with higher beta using PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_polished = _pgd_optimize(best_h, beta=100.0, n_steps=5000, lr_init=0.005)
        _update_best(h_polished)
        
    # 7. Final ultra-precise polish with SLSQP
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_final = _polish(best_h, beta=200.0, maxiter=1000, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)
        
    # 8. Very final PGD polish with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_very_final = _pgd_optimize(best_h, beta=500.0, n_steps=3000, lr_init=0.002)
        _update_best(h_very_final)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
