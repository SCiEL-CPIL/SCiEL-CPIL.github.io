"""Erdos minimum overlap: LP structural solve + spectral subgradient refinement."""
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 100
DEADLINE = 145  # safety margin before 150s kill


def _max_overlap(h):
    """Compute maximum overlap of h with shifted (1-h) using FFT."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    return np.max(corr)


def _soft_min_overlap(h, alpha=100.0):
    """Soft-min approximation of max overlap for smooth optimization."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    m = np.max(corr)
    shifted = (corr - m) * alpha
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted)) / N) / alpha
    return log_sum_exp


def _soft_min_overlap_grad(h, alpha=100.0):
    """Gradient of soft-min overlap objective."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    m = np.max(corr)
    shifted = (corr - m) * alpha
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad_corr = np.zeros((N, N))
    indices = np.arange(N)
    for k in range(N):
        term1 = 1.0 - h[(indices - k) % N]
        term2 = h[(indices + k) % N]
        grad_corr[k, :] = term1 - term2
    
    grad = weights @ grad_corr
    return grad


def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    diff = target - np.sum(h)
    
    for _ in range(50):
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.minimum(h[mask] + step, 1.0)
        else:
            mask = h > 1e-12
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.maximum(h[mask] + step, 0.0)
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        interior = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(interior):
            h[interior] += diff / np.sum(interior)
            h = np.clip(h, 0.0, 1.0)
    return h


def _lp_solve(n_steps):
    """Solve the LP formulation for the optimal step heights."""
    step_width = 2.0 / n_steps
    
    # Candidate shift values where the overlap function is piecewise linear
    s_candidates = set()
    s_candidates.add(0.0)
    s_candidates.add(1.0)
    s_candidates.add(2.0)
    for k in range(n_steps + 1):
        s1 = 2.0 * k / n_steps
        s2 = 1.0 - 2.0 * k / n_steps
        if 0.0 <= s1 <= 2.0:
            s_candidates.add(round(s1, 12))
        if 0.0 <= s2 <= 2.0:
            s_candidates.add(round(s2, 12))
    
    s_vals = sorted(s_candidates)
    num_constraints = len(s_vals)
    
    A_ub = np.zeros((num_constraints, n_steps + 1))
    b_ub = np.zeros(num_constraints)
    
    for idx, s in enumerate(s_vals):
        d = 0.0
        for i in range(n_steps):
            step_start = 2.0 * i / n_steps
            step_end = 2.0 * (i + 1) / n_steps
            
            if s <= 1.0:
                lo1, hi1 = max(0.0, step_start), min(1.0 - s, step_end)
                lo2, hi2 = max(1.0 + s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                w_i = m1 - m2
                A_ub[idx, i] = w_i
                d += m2
            else:
                lo1, hi1 = max(0.0, step_start), min(s - 1.0, step_end)
                lo2, hi2 = max(2.0 - s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                w_i = -m1 + m2
                A_ub[idx, i] = w_i
                d += m1
        
        A_ub[idx, n_steps] = -1.0
        b_ub[idx] = -d
    
    A_eq = np.zeros((1, n_steps + 1))
    A_eq[0, :n_steps] = 1.0
    b_eq = np.array([n_steps / 2.0])
    
    bounds = [(0, 1)] * n_steps + [(0, None)]
    c = np.zeros(n_steps + 1)
    c[n_steps] = 1.0
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        if result.success:
            h_val = result.x[:n_steps]
            h_val = np.clip(h_val, 0.0, 1.0)
            diff = n_steps / 2.0 - np.sum(h_val)
            if abs(diff) > 1e-12:
                h_val += diff / n_steps
                h_val = np.clip(h_val, 0.0, 1.0)
            return h_val
    except Exception:
        pass
    return None


def _subgradient_descent(h0, max_iter=2000, alpha_init=0.01, time_limit=None, t_start=None):
    """Projected subgradient descent for soft-min overlap."""
    if t_start is None:
        t_start = time.time()
    
    h = np.asarray(h0, dtype=np.float64).copy()
    h = _project(h)
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    alpha = alpha_init
    soft_alpha = 100.0
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - t_start > time_limit:
            break
        
        obj = _soft_min_overlap(h, soft_alpha)
        grad = _soft_min_overlap_grad(h, soft_alpha)
        
        h_new = h - alpha * grad
        h_new = _project(h_new)
        
        val = _max_overlap(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        
        h = h_new
        
        if it % 100 == 0 and it > 0:
            alpha *= 0.9
        soft_alpha = min(soft_alpha * 1.001, 500.0)
        
        if it > 100 and abs(_soft_min_overlap(h, soft_alpha) - obj) < 1e-12:
            break
    
    return best_h, best_val


def _bfgs_optimize(h0, max_iter=500, time_limit=None, t_start=None):
    """BFGS optimization using scipy."""
    if t_start is None:
        t_start = time.time()
    
    h0 = np.asarray(h0, dtype=np.float64).copy()
    h0 = _project(h0)
    
    def obj_and_grad(x):
        if time_limit is not None and time.time() - t_start > time_limit:
            return _soft_min_overlap(x), np.zeros_like(x)
        f = _soft_min_overlap(x, alpha=150.0)
        g = _soft_min_overlap_grad(x, alpha=150.0)
        return f, g
    
    try:
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            h0,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': max_iter, 'maxfun': 5000}
        )
        h_opt = _project(result.x)
        val = _max_overlap(h_opt)
        return h_opt, val
    except Exception:
        return h0, _max_overlap(h0)


def _spectral_init():
    """Generate a spectral-shaped initial guess."""
    rng = np.random.default_rng(12345)
    mag = np.ones(N)
    mag[0] = N / 2.0
    phases = rng.uniform(-np.pi, np.pi, N)
    H = mag * np.exp(1j * phases)
    for k in range(1, N // 2 + 1):
        H[N - k] = np.conj(H[k])
    h = np.fft.ifft(H).real
    h = h - np.mean(h) + 0.5
    h = np.clip(h, 0, 1)
    return _project(h)


def construct():
    t0 = time.time()
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: LP structural solve (the key improvement from inspiration 1)
    # Try multiple resolutions; the LP gives a globally optimal solution for that resolution
    for n_steps in [100, 50, 200]:
        if time.time() - t0 > DEADLINE - 40:
            break
        h_lp = _lp_solve(n_steps)
        if h_lp is not None:
            if n_steps != N:
                # Interpolate to N if needed
                if n_steps < N:
                    x_old = np.linspace(0, 2, n_steps + 1)
                    x_new = np.linspace(0, 2, N + 1)
                    h_interp = np.interp(x_new, x_old, np.concatenate([h_lp, [h_lp[-1]]]))
                    h_lp = h_interp[:-1]
                else:
                    # Downsample
                    x_old = np.linspace(0, 2, n_steps + 1)
                    x_new = np.linspace(0, 2, N + 1)
                    h_interp = np.interp(x_new, x_old, np.concatenate([h_lp, [h_lp[-1]]]))
                    h_lp = h_interp[:-1]
                h_lp = _project(h_lp)
            
            val = _max_overlap(h_lp)
            if val < best_val:
                best_val = val
                best_h = h_lp.copy()
    
    # Phase 2: Spectral + random initializations as fallbacks
    if time.time() - t0 < DEADLINE - 30:
        inits = []
        
        # Uniform baseline
        inits.append(np.full(N, 0.5))
        
        # Spectral shaping
        inits.append(_spectral_init())
        
        # Sinusoidal patterns
        for freq in [1, 2, 3, 4, 5]:
            for A in [0.2, 0.3, 0.4, 0.5]:
                h = 0.5 + A * np.sin(2 * np.pi * freq * np.arange(N) / N)
                inits.append(_project(np.clip(h, 0, 1)))
        
        # Random
        rng = np.random.default_rng(42)
        for _ in range(10):
            h = rng.uniform(0, 1, N)
            inits.append(_project(h))
        
        for h0 in inits:
            val = _max_overlap(h0)
            if val < best_val:
                best_val = val
                best_h = h0.copy()
            
            if time.time() - t0 > DEADLINE - 30:
                break
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_val = _max_overlap(best_h)
    
    # Phase 3: Subgradient refinement from best
    remaining = DEADLINE - 15 - (time.time() - t0)
    if remaining > 5:
        h_opt, val_opt = _subgradient_descent(best_h, max_iter=1500, time_limit=remaining, t_start=t0)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 4: BFGS polish
    remaining = DEADLINE - 5 - (time.time() - t0)
    if remaining > 3:
        h_opt, val_opt = _bfgs_optimize(best_h, max_iter=300, time_limit=remaining, t_start=t0)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 5: Try diverse starts with quick optimization if time remains
    diverse_starts = [
        np.array([(2*i + 1) / (2*N) for i in range(N)]),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.4 * np.cos(2 * np.pi * 2 * np.arange(N) / N),
    ]
    
    for h_start in diverse_starts:
        if time.time() - t0 > DEADLINE - 8:
            break
        h_start = _project(h_start)
        val_start = _max_overlap(h_start)
        if val_start < best_val + 0.5:
            remaining = DEADLINE - 5 - (time.time() - t0)
            if remaining > 2:
                h_opt, val_opt = _subgradient_descent(h_start, max_iter=500, time_limit=remaining, t_start=t0)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
    
    # Final BFGS polish
    remaining = DEADLINE - 2 - (time.time() - t0)
    if remaining > 1:
        h_opt, val_opt = _bfgs_optimize(best_h, max_iter=100, time_limit=remaining, t_start=t0)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    final_h = _project(best_h)
    return final_h
