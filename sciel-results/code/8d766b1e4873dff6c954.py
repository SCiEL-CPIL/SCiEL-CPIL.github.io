"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
FFT-accelerated overlap, exact Euclidean projection onto box+sum polytope,
Nesterov accelerated gradient on log-sum-exp smooth-max, and SLSQP polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_DEADLINE = time.time() + 142


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp."""
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


def _project(h):
    """Exact Euclidean projection onto {h: 0<=h_i<=1, sum(h)=N/2}."""
    h = np.asarray(h, dtype=np.float64)
    # Bisection on threshold tau: find tau such that sum(clip(h_i - tau, 0, 1)) = target
    lo = h.min() - 1.0
    hi = h.max() + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        projected = np.clip(h - mid, 0.0, 1.0)
        s = projected.sum()
        if s > _TARGET:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2.0
    result = np.clip(h - mid, 0.0, 1.0)
    # Fix tiny numerical drift
    diff = _TARGET - result.sum()
    if abs(diff) > 1e-14:
        # Adjust the element closest to 0.5 (least impact on boundary)
        idx = np.argmin(np.abs(result - 0.5))
        result[idx] = np.clip(result[idx] + diff, 0.0, 1.0)
    return result


def _accelerated_smooth_max(h0, beta, max_iter=500, tol=1e-10):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h)
    
    # Nesterov momentum
    h_prev = h.copy()
    t_k = 1.0
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(h, beta)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project onto {h: 0 <= h_i <= 1, sum(h) = N/2}
        h_new = _project(h_new)
        
        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _project(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        # Adaptive learning rate
        if it % 50 == 49:
            lr = max(0.001, min(0.1, lr * 0.95))
    
    return h, _score(h)


def _coordinate_descent_polish(h0, max_iter=500):
    """Coordinate descent: adjust pairs of coordinates to reduce max overlap.
    
    For each coordinate i, the overlap C_k is quadratic in h. We find the
    optimal value for h_i (with h_j adjusted to maintain sum) that minimizes
    the resulting max overlap.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    n = len(h)
    best_s = _score(h)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 8:
            break
        
        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - 8:
                break
            
            # Try adjusting h[i] and h[(i+1) % n] to maintain sum
            j = (i + 1) % n
            if i == j:
                continue
            
            old_hi = h[i]
            old_hj = h[j]
            
            # Try a few perturbation sizes
            for delta in [0.01, 0.02, 0.05, -0.01, -0.02, -0.05]:
                new_hi = np.clip(old_hi + delta, 0.0, 1.0)
                new_hj = np.clip(old_hj - delta, 0.0, 1.0)
                
                # Check if sum is preserved (approximately)
                if abs((new_hi + new_hj) - (old_hi + old_hj)) > 1e-10:
                    continue
                
                h[i] = new_hi
                h[j] = new_hj
                s = _score(h)
                
                if s < best_s - 1e-12:
                    best_s = s
                    improved = True
                else:
                    h[i] = old_hi
                    h[j] = old_hj
            
            if not improved and it > 100:
                # Try smaller steps
                for delta in [0.005, 0.01, -0.005, -0.01]:
                    new_hi = np.clip(old_hi + delta, 0.0, 1.0)
                    new_hj = np.clip(old_hj - delta, 0.0, 1.0)
                    
                    if abs((new_hi + new_hj) - (old_hi + old_hj)) > 1e-10:
                        continue
                    
                    h[i] = new_hi
                    h[j] = new_hj
                    s = _score(h)
                    
                    if s < best_s - 1e-12:
                        best_s = s
                        improved = True
                    else:
                        h[i] = old_hi
                        h[j] = old_hj
        
        if not improved:
            break
    
    h = _project(h)
    return h, _score(h)


def _slsqp_polish(h0, beta=100.0, maxiter=300):
    """SLSQP polish: minimize smooth_max with high beta, box+sum constraints."""
    if time.time() > _DEADLINE - 8:
        return h0, _score(h0)
    
    def objective(x):
        x = np.clip(x, 0.0, 1.0)
        val, _ = _smooth_max_val_grad(x, beta)
        return val
    
    def gradient(x):
        x = np.clip(x, 0.0, 1.0)
        _, g = _smooth_max_val_grad(x, beta)
        return g
    
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - _TARGET,
                    'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(objective, h0, jac=gradient, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14})
        if res.success or res.fun < _score(h0):
            h_res = _project(res.x)
            return h_res, _score(h_res)
    except Exception:
        pass
    return h0, _score(h0)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(12345)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    i = np.arange(N)
    
    # Diverse initial guesses
    starts = []
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_project(h_cos))
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_project(h_sin))
    
    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_project(h_2f))
    
    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_project(h_saw))
    
    # Random
    for _ in range(5):
        h_r = np.random.uniform(0, 1, N)
        starts.append(_project(h_r))
    
    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)
    
    # Multi-phase beta schedule: coarse to fine
    betas = [20.0, 40.0, 80.0, 150.0, 300.0, 600.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 20:
            break
        
        # Run accelerated gradient from best
        h_candidate, s_candidate = _accelerated_smooth_max(
            best_h, beta, max_iter=500, tol=1e-12
        )
        if s_candidate < best_score:
            _update_best(h_candidate)
        
        # Try from perturbed starts
        for _ in range(3):
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = _project(h_pert)
            h_cand2, s_cand2 = _accelerated_smooth_max(
                h_pert, beta, max_iter=300, tol=1e-12
            )
            if s_cand2 < best_score:
                _update_best(h_cand2)
    
    # Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_cd, s_cd = _coordinate_descent_polish(best_h, max_iter=300)
        if s_cd < best_score:
            _update_best(h_cd)
    
    # SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 12:
        h_slq, s_slq = _slsqp_polish(best_h, beta=100.0, maxiter=300)
        if s_slq < best_score:
            _update_best(h_slq)
    
    # Final high-beta accelerated gradient polish
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_final, s_final = _accelerated_smooth_max(best_h, beta=1000.0, max_iter=600, tol=1e-14)
        if s_final < best_score:
            _update_best(h_final)
    
    # Another SLSQP pass with higher beta
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_slq2, s_slq2 = _slsqp_polish(best_h, beta=200.0, maxiter=200)
        if s_slq2 < best_score:
            _update_best(h_slq2)
    
    # Final coordinate descent
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_cd2, s_cd2 = _coordinate_descent_polish(best_h, max_iter=200)
        if s_cd2 < best_score:
            _update_best(h_cd2)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h)
    return best_h
