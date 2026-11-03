"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses projected gradient descent with log-sum-exp smoothing, SLSQP polish,
and multiple diverse initializations for robust basin exploration."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


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


def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to smooth_max(h) <= t,
    with x = [h_0, ..., h_{N-1}, t].
    """
    h = x[:N]
    t = x[N]
    val, _ = _smooth_max_grad(h, beta)
    return t + 0.01 * max(0, val - t)  # Penalize constraint violation


def _epigraph_grad(x, beta):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    val, grad_h = _smooth_max_grad(h, beta)
    
    grad = np.zeros(len(x))
    if val > t:
        grad[:N] = grad_h * 0.01
        grad[N] = 1.0 + 0.01
    else:
        grad[:N] = grad_h * 0.0
        grad[N] = 1.0
    
    return grad


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


def _project_sum_box(h):
    """
    Project h onto the set {x : sum(x) = N/2, 0 <= x_i <= 1}.
    Uses iterative water-filling style projection.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        # Adjust only elements not at bounds
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if not np.any(mask):
            # If all at bounds, distribute among those closest to target
            idx = np.argsort(np.abs(h - 0.5))[:10]
            mask = np.zeros(N, dtype=bool)
            mask[idx] = True
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    
    return h


def _pgd_optimize(h0, beta, max_iter=5000, lr_schedule='adaptive', tol=1e-10):
    """
    Projected gradient descent with adaptive step size.
    Minimizes smooth_max(h) subject to sum(h)=N/2, 0<=h<=1.
    """
    h = _fix_sum(h0.copy())
    best_h = h.copy()
    best_val = _score(h)
    
    # Initial learning rate
    lr = 0.01
    prev_val = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        val, grad = _smooth_max_grad(h, beta)
        
        # Adaptive step size
        if lr_schedule == 'adaptive':
            # Use Barzilai-Borwein style or simple backtracking
            if it > 0 and prev_val > val:
                # Converging, can increase lr slightly
                lr = min(lr * 1.05, 0.05)
            else:
                # Not converging, decrease lr
                lr = max(lr * 0.8, 0.001)
        else:
            lr = 0.01
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project onto feasible set
        h_new = _project_sum_box(h_new)
        
        new_val = _smooth_max_value(h_new, beta)
        
        if new_val < prev_val:
            h = h_new
            prev_val = new_val
            # Update best
            s = _score(h)
            if s < best_val:
                best_val = s
                best_h = h.copy()
        else:
            # Line search: reduce step size
            lr *= 0.5
            h_new = h - lr * grad
            h_new = _project_sum_box(h_new)
            new_val = _smooth_max_value(h_new, beta)
            if new_val < prev_val:
                h = h_new
                prev_val = new_val
                s = _score(h)
                if s < best_val:
                    best_val = s
                    best_h = h.copy()
        
        # Convergence check
        if it > 100 and abs(prev_val - _smooth_max_value(h, beta)) < tol:
            break
    
    return best_h


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
    
    # 1. Generate diverse initializations
    i = np.arange(N)
    initializations = []
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_cos))
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_sin))
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    initializations.append(_fix_sum(h_alt))
    
    # Block patterns
    for block_size in [5, 10, 20]:
        h_block = np.zeros(N)
        for start in range(0, N, block_size * 2):
            h_block[start:start+block_size] = 0.8
            h_block[start+block_size:start+2*block_size] = 0.2
        initializations.append(_fix_sum(h_block))
    
    # Random
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        initializations.append(_fix_sum(h_rand))
    
    # 2. Run PGD from each initialization
    for h_init in initializations:
        if time.time() > _DEADLINE - 30:
            break
        h_pgd = _pgd_optimize(h_init, beta=50.0, max_iter=3000)
        _update_best(h_pgd)
    
    # 3. SLSQP polish on best
    if best_h is not None and time.time() < _DEADLINE - 20:
        h_slsqp = _run_slsqp(best_h, beta=50.0, maxiter=200, ftol=1e-10, gtol=1e-8)
        _update_best(h_slsqp)
    
    # 4. Higher beta PGD polish
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_pgd_high = _pgd_optimize(best_h, beta=100.0, max_iter=2000)
        _update_best(h_pgd_high)
    
    # 5. SLSQP with higher beta
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_slsqp_high = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-12, gtol=1e-10)
        _update_best(h_slsqp_high)
    
    # 6. Final ultra-precise polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_final = _polish(best_h, beta=200.0, maxiter=500, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
