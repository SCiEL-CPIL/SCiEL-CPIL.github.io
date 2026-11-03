import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145.0  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum h_i * h_{i-k} (circular)."""
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
    """Return the maximum overlap value."""
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


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
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


def _accelerated_smooth_max(h0, beta, max_iter=400):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    
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
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)
        
        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = np.clip(h_next, 0.0, 1.0)
        h_next = _fix_sum(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        # Adaptive learning rate
        if it % 50 == 49:
            lr = max(0.001, min(0.1, lr * 0.95))
    
    return h, _score(h)


def _scipy_refine(h0, beta=50.0):
    """Use scipy L-BFGS-B to refine h with smooth-max objective."""
    h0 = _fix_sum(h0)
    
    def obj_and_grad(x):
        val, grad = _smooth_max_val_grad(x, beta)
        return val, grad
    
    # Constraints: sum(h) = N/2
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(obj_and_grad, h0, method='L-BFGS-B', jac=True,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        h_refined = _fix_sum(res.x)
        return h_refined, _score(h_refined)
    except Exception:
        return h0, _score(h0)


def _anneal_smooth_max(h0, beta_start=20.0, beta_end=200.0, total_iter=600):
    """Anneal beta from low to high for better convergence."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    
    n_betas = 10
    betas = np.geomspace(beta_start, beta_end, n_betas)
    iters_per_beta = total_iter // n_betas
    
    for beta in betas:
        if time.time() > _DEADLINE - 10:
            break
        h, _ = _accelerated_smooth_max(h, beta, max_iter=iters_per_beta)
    
    return h, _score(h)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(12345)
    
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
    
    # Diverse initial guesses
    starts = []
    
    # Constant 0.5
    starts.append(np.full(N, 0.5))
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_cos))
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_sin))
    
    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))
    
    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))
    
    # Block pattern: alternating high/low
    h_block = np.where(i % 2 == 0, 0.8, 0.2)
    starts.append(_fix_sum(h_block))
    
    # Square wave (4-cycle)
    h_sq = np.where((i // 25) % 2 == 0, 0.7, 0.3)
    starts.append(_fix_sum(h_sq))
    
    # High frequency cosine
    h_hf = 0.5 + 0.4 * np.cos(2.0 * np.pi * 5 * i / N)
    starts.append(_fix_sum(h_hf))
    
    # Random starts
    for _ in range(10):
        h_r = np.random.uniform(0, 1, N)
        starts.append(_fix_sum(h_r))
    
    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)
    
    # Phase 1: Annealed optimization from best start
    if best_h is not None and time.time() < _DEADLINE - 30:
        h_ann, s_ann = _anneal_smooth_max(best_h, beta_start=20.0, beta_end=200.0, total_iter=500)
        if s_ann < best_score:
            _update_best(h_ann)
    
    # Phase 2: Bisection on t with accelerated gradient feasibility
    t_low = 0.0
    t_high = best_score
    
    for bisect_iter in range(15):
        if time.time() > _DEADLINE - 20:
            break
        
        t_mid = (t_low + t_high) / 2.0
        beta = 50.0 + bisect_iter * 10
        
        # Try to find h with smooth_max(h, beta) <= t_mid
        # Use accelerated gradient from best_h
        if best_h is not None:
            h_candidate, s_candidate = _accelerated_smooth_max(
                best_h, beta, max_iter=250
            )
            if s_candidate < best_score:
                _update_best(h_candidate)
            
            # Also try from a perturbed start
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_candidate2, s_candidate2 = _accelerated_smooth_max(
                h_pert, beta, max_iter=150
            )
            if s_candidate2 < best_score:
                _update_best(h_candidate2)
        
        # Check if feasible: if best_score <= t_mid, go lower
        if best_score <= t_mid + 1e-6:
            t_high = t_mid
        else:
            t_low = t_mid
        
        # If gap is small, break
        if t_high - t_low < 1e-8:
            break
    
    # Phase 3: Final polish with scipy at high beta
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_final, s_final = _scipy_refine(best_h, beta=150.0)
        if s_final < best_score:
            _update_best(h_final)
    
    # Phase 4: Additional scipy refinements from different starts
    if time.time() < _DEADLINE - 15:
        for h0 in starts[:8]:
            if time.time() > _DEADLINE - 8:
                break
            h_ref, s_ref = _scipy_refine(h0, beta=80.0)
            if s_ref < best_score:
                _update_best(h_ref)
    
    # Phase 5: Very high beta polish if time permits
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_polish, s_polish = _scipy_refine(best_h, beta=300.0)
        if s_polish < best_score:
            _update_best(h_polish)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
