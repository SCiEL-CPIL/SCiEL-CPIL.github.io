"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses epigraph formulation with SLSQP, refined with higher precision polishes.
Combines cosine initialization with random restarts and adaptive beta tuning."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


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


def _trig_basis(N, K):
    """
    Build a trigonometric basis matrix of shape (N, K).
    Column 0: constant (DC)
    Columns 1..K-1: cos(2*pi*m*i/N) and sin(2*pi*m*i/N) for m=1,...
    """
    i = np.arange(N)
    cols = [np.ones(N)]  # DC component
    m = 1
    while len(cols) < K:
        cols.append(np.cos(2.0 * np.pi * m * i / N))
        if len(cols) < K:
            cols.append(np.sin(2.0 * np.pi * m * i / N))
        m += 1
    return np.column_stack(cols[:K])


def _optimize_trig_basis(K, beta, h_init=None, maxiter=200, ftol=1e-10, gtol=1e-8):
    """
    Optimize in the trigonometric basis space.
    h = basis @ coeffs, with constraint sum(h) = N/2.
    The DC coefficient is determined by the sum constraint.
    """
    basis = _trig_basis(N, K)
    # basis[:, 0] is all ones, so sum(h) = N * coeffs[0] + sum_{j>=1} basis[j] * coeffs[j]
    # We need sum(h) = N/2, so coeffs[0] = (N/2 - sum_{j>=1} basis[j]*coeffs[j]) / N
    
    def coeffs_to_h(coeffs):
        """Convert coefficients to h vector, enforcing sum constraint."""
        h = np.zeros(N)
        # Sum of non-DC components
        non_dc_sum = basis[1:] @ coeffs[1:]
        # DC coefficient to satisfy sum = N/2
        dc_coeff = (N / 2.0 - non_dc_sum) / N
        h = dc_coeff * basis[:, 0] + basis[:, 1:] @ coeffs[1:]
        return h
    
    def trig_objective(c):
        """Objective: smooth max of overlap values."""
        h = coeffs_to_h(c)
        val, _ = _smooth_max_grad(h, beta)
        return val
    
    def trig_grad(c):
        """Gradient of objective w.r.t. coefficients."""
        h = coeffs_to_h(c)
        val, grad_h = _smooth_max_grad(h, beta)
        # Chain rule: d(val)/dc_j = grad_h @ d(h)/dc_j
        # d(h)/dc_0 = basis[:, 0] / N (from the DC constraint)
        # Actually, let's be more careful.
        # h = (N/2 - basis[1:]@c[1:]) / N * basis[:,0] + basis[:,1:] @ c[1:]
        # dh/dc_0 = 0 (c_0 is not free, it's determined)
        # Wait, in this formulation, we only optimize c[1:] (the non-DC coefficients).
        # c[0] is determined by the constraint.
        # So the free variables are c[1:].
        # Let's redefine: we optimize only the non-DC coefficients.
        return val, grad_h
    
    # We optimize only the non-DC coefficients (c[1:])
    n_free = K - 1
    
    def obj_free(c_free):
        """Objective with only free coefficients."""
        c = np.zeros(K)
        c[1:] = c_free
        h = coeffs_to_h(c)
        val, _ = _smooth_max_grad(h, beta)
        return val
    
    def jac_free(c_free):
        """Gradient w.r.t. free coefficients."""
        c = np.zeros(K)
        c[1:] = c_free
        h = coeffs_to_h(c)
        val, grad_h = _smooth_max_grad(h, beta)
        
        # h = dc_coeff * ones + basis[:,1:] @ c_free
        # dc_coeff = (N/2 - basis[1:] @ c_free) / N
        # dh/dc_free_j = -ones * basis[j] / N + basis[:, j]  (for j = 1..K-1)
        
        grad_c = np.zeros(n_free)
        for j in range(n_free):
            # basis column index j+1
            col = basis[:, j + 1]
            dh_dc = -col / N + col  # = col * (1 - 1/N)
            grad_c[j] = np.dot(grad_h, dh_dc)
        
        return val, grad_c
    
    # Initial coefficients: project h_init onto the basis
    if h_init is not None:
        h_init = np.asarray(h_init, dtype=np.float64)
        # Least squares projection
        c_full, _, _, _ = np.linalg.lstsq(basis, h_init, rcond=None)
        c_free_init = c_full[1:]
    else:
        c_free_init = np.zeros(n_free)
    
    # Bounds for free coefficients: we need h in [0,1]
    # This is complex, so we'll use penalty or just let SLSQP handle it
    # For now, use moderate bounds
    bounds = [(-1.0, 1.0)] * n_free
    
    try:
        res = minimize(obj_free, c_free_init.copy(), jac=jac_free, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        c_free_opt = res.x
        c_opt = np.zeros(K)
        c_opt[1:] = c_free_opt
        h_opt = coeffs_to_h(c_opt)
        return h_opt
    except Exception:
        return h_init if h_init is not None else np.full(N, 0.5)


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
    
    # 1. Cosine start
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Trigonometric basis optimization with increasing K
    if time.time() - t0 < _DEADLINE - 60:
        for K in [5, 8, 12, 16]:
            if time.time() - t0 > _DEADLINE - 30:
                break
            h_trig = _optimize_trig_basis(K, beta=50.0, h_init=best_h, 
                                          maxiter=200, ftol=1e-10, gtol=1e-8)
            _update_best(h_trig)
    
    # 3. Primary optimization with moderate beta (full space)
    if time.time() - t0 < _DEADLINE - 35:
        h_opt = _run_slsqp(best_h, beta=50.0, maxiter=300, ftol=1e-10, gtol=1e-8)
        _update_best(h_opt)
        
    # 4. Random restarts in trigonometric basis
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            # Random coefficients for trig basis
            K = 12
            basis = _trig_basis(N, K)
            c_free_rand = np.random.uniform(-0.3, 0.3, K - 1)
            c_rand = np.zeros(K)
            c_rand[1:] = c_free_rand
            h_rand = (N / 2.0 - basis[1:] @ c_rand[1:]) / N * basis[:, 0] + basis[:, 1:] @ c_rand[1:]
            h_rand = _fix_sum(h_rand)
            h_opt = _run_slsqp(h_rand, beta=50.0, maxiter=150, ftol=1e-9, gtol=1e-7)
            _update_best(h_opt)
            
    # 5. Polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_polished = _polish(best_h, beta=100.0, maxiter=500, ftol=1e-12, gtol=1e-10)
        _update_best(h_polished)
        
    # 6. Final ultra-precise polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_final = _polish(best_h, beta=200.0, maxiter=1000, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
