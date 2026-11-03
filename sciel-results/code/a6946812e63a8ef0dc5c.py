import numpy as np
import time

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _project_to_feasible(h):
    """
    Project h onto the feasible set: sum(h) = N/2, 0 <= h_i <= 1.
    Uses iterative projection (alternating between sum constraint and box constraint).
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # Iterate to satisfy both constraints
    for _ in range(20):
        # Enforce sum constraint
        diff = target - np.sum(h)
        if abs(diff) > 1e-12:
            # Only adjust elements that are not at bounds
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
            else:
                # All at bounds, distribute among all
                h += diff / N
        # Enforce box constraint
        h = np.clip(h, 0.0, 1.0)
    
    # Final exact sum adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        # Find elements with most room
        if diff > 0:
            order = np.argsort(-h)  # Most room first (closest to 0)
        else:
            order = np.argsort(h)   # Most room first (closest to 1)
        for idx in order:
            room = (1.0 - h[idx]) if diff > 0 else (h[idx] - 0.0)
            if room > 1e-15:
                adjust = min(abs(diff), room)
                h[idx] += adjust if diff > 0 else -adjust
                diff -= adjust if diff > 0 else -adjust
                if abs(diff) < 1e-15:
                    break
    return h


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Vectorized gradient computation using FFT.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
    return val, grad


def _pgd_optimize(h0, beta, n_iter=500, lr=None, time_limit=30.0):
    """
    Projected Gradient Descent with adaptive learning rate.
    Returns the best h found.
    """
    h = _project_to_feasible(h0.copy())
    best_h = h.copy()
    best_val = _score(h)
    
    if lr is None:
        # Estimate Lipschitz constant for gradient
        # Gradient involves convolutions, so Lipschitz constant is bounded by ~4/N * max_w * max_h
        # Use a conservative initial LR
        lr = 0.5 / (beta * N)
    
    t_start = time.time()
    
    for i in range(n_iter):
        if time.time() - t_start > time_limit:
            break
        
        val, grad = _smooth_max_grad(h, beta)
        
        # Update
        h_new = h - lr * grad
        
        # Project to feasible set
        h_new = _project_to_feasible(h_new)
        
        # Check improvement
        new_val = _score(h_new)
        if new_val < best_val - 1e-12:
            best_val = new_val
            best_h = h_new.copy()
            h = h_new
            # Slightly increase LR if improving
            lr *= 1.01
        else:
            # Decrease LR
            lr *= 0.8
            h = h_new  # Keep the projected point even if not improving
        
        # Adaptive LR: if stuck, try larger step
        if i % 50 == 0 and i > 0:
            # Check if we've been stuck
            recent_improvement = (best_val - val) > 1e-8
            if not recent_improvement and lr < 0.1:
                lr *= 1.5
    
    return best_h, best_val


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    return _project_to_feasible(h)


def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to cs <= t, sum(h) = N/2, 0 <= h <= 1.
    x = [h_0, ..., h_{N-1}, t]
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    
    # Smooth constraint violation: sum of max(0, cs - t)^2
    violations = np.maximum(cs - t, 0)
    penalty = np.sum(violations ** 2)
    
    # Smooth max approximation for better gradient
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    soft_max = m + np.log(sum_exp) / beta
    
    # Objective: t + penalty + small regularization
    return t + penalty + 1e-6 * (t - soft_max) ** 2


def _epigraph_gradient(x, beta):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    
    # Gradient of penalty term
    violations = np.maximum(cs - t, 0)
    active = violations > 0
    
    # d/dh of penalty: -2 * sum_{k: cs_k > t} (cs_k - t) * dcs_k/dh
    # dcs_k/dh_j = -(2/N) * (h_{j-k} + h_{j+k}) for valid indices
    n = N
    n_fft = 2 * n
    w = np.where(active, cs - t, 0)
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    grad_h = -2.0 * (-(2.0 / n) * (conv1 + conv2))
    
    # Gradient of t term: 1 + 2*1e-6*(t - soft_max)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    soft_max = m + np.log(sum_exp) / beta
    grad_t = 1.0 + 2e-6 * (t - soft_max)
    
    # Add penalty gradient contribution to grad_t
    grad_t += 2.0 * np.sum(violations)
    
    grad = np.zeros(N + 1)
    grad[:N] = grad_h
    grad[N] = grad_t
    
    return grad


def _pgd_epigraph_optimize(h0, t0, beta, n_iter=300, lr_h=0.01, lr_t=0.01, time_limit=20.0):
    """
    Projected Gradient Descent for epigraph formulation.
    """
    h = _project_to_feasible(h0.copy())
    t = t0
    best_h = h.copy()
    best_t = t
    best_val = _score(h)
    
    t_start = time.time()
    
    for i in range(n_iter):
        if time.time() - t_start > time_limit:
            break
        
        cs = _linear_overlap_values(h)
        
        # Gradient for h
        violations = np.maximum(cs - t, 0)
        active = violations > 0
        w = np.where(active, cs - t, 0)
        
        n = N
        n_fft = 2 * n
        W = np.fft.rfft(w, n=n_fft)
        H = np.fft.rfft(h, n=n_fft)
        
        conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
        v = h[::-1]
        V = np.fft.rfft(v, n=n_fft)
        conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
        conv2 = conv2_rev[n-1::-1]
        
        grad_h = -2.0 * (-(2.0 / n) * (conv1 + conv2))
        
        # Gradient for t
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        soft_max = m + np.log(sum_exp) / beta
        grad_t = 1.0 + 2e-6 * (t - soft_max) + 2.0 * np.sum(violations)
        
        # Update
        h_new = h - lr_h * grad_h
        h_new = _project_to_feasible(h_new)
        t_new = t - lr_t * grad_t
        t_new = max(0.0, t_new)
        
        # Check improvement
        new_val = _score(h_new)
        if new_val < best_val - 1e-12:
            best_val = new_val
            best_h = h_new.copy()
            best_t = t_new
            h = h_new
            t = t_new
            lr_h *= 1.01
            lr_t *= 1.01
        else:
            lr_h *= 0.8
            lr_t *= 0.8
            h = h_new
            t = t_new
    
    return best_h, best_t, best_val


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
    
    # 1. Multiple Analytical Starts
    i = np.arange(N)
    
    # Cosine wave (fundamental)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # Higher harmonic
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    h_cos2 = _fix_sum(h_cos2)
    _update_best(h_cos2)
    
    # Triangular wave
    h_tri = 0.5 + 0.5 * np.abs(2.0 * (i % N) / N - 1.0)
    h_tri = _fix_sum(h_tri)
    _update_best(h_tri)
    
    # 2. Multi-stage PGD optimization with adaptive beta
    def _run_pgd(h0, beta, n_iter=300, lr=None, time_limit=15.0):
        nonlocal best_h, best_score
        h_res, val_res = _pgd_optimize(h0, beta, n_iter=n_iter, lr=lr, time_limit=time_limit)
        _update_best(h_res)
    
    def _run_pgd_epigraph(h0, beta, n_iter=200, lr_h=0.01, lr_t=0.01, time_limit=10.0):
        nonlocal best_h, best_score
        cs = _linear_overlap_values(h0)
        t0_init = np.max(cs) + 0.1
        h_res, t_res, val_res = _pgd_epigraph_optimize(h0, t0_init, beta, n_iter=n_iter, lr_h=lr_h, lr_t=lr_t, time_limit=time_limit)
        _update_best(h_res)
    
    # Phase 1: Coarse PGD from best analytical start
    if time.time() - t0 < _DEADLINE - 40:
        _run_pgd(best_h, beta=20.0, n_iter=400, time_limit=15.0)
        
    # Phase 2: Medium PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        _run_pgd(best_h, beta=50.0, n_iter=500, time_limit=15.0)
        
    # Phase 3: Try epigraph PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        _run_pgd_epigraph(best_h, beta=50.0, n_iter=300, time_limit=10.0)
        
    # Phase 4: Fine PGD with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        _run_pgd(best_h, beta=100.0, n_iter=600, time_limit=12.0)
        
    # Phase 5: Very fine PGD with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        _run_pgd(best_h, beta=200.0, n_iter=800, time_limit=10.0)
        
    # Phase 6: Final polish with extremely high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        _run_pgd(best_h, beta=500.0, n_iter=1000, time_limit=8.0)
        
    # Phase 7: Structured perturbations with PGD refinement
    if time.time() - t0 < _DEADLINE - 10:
        for scale in [0.03, 0.015, 0.008]:
            if time.time() - t0 > _DEADLINE - 5:
                break
            for _ in range(4):
                if time.time() - t0 > _DEADLINE - 3:
                    break
                noise = np.random.normal(0, scale, N)
                h_pert = best_h + noise
                h_pert = _fix_sum(h_pert)
                _update_best(h_pert)
                if time.time() - t0 < _DEADLINE - 3:
                    _run_pgd(h_pert, beta=100.0, n_iter=200, time_limit=5.0)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
