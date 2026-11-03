"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses spectral representation, Nesterov accelerated projected gradient,
and high-precision SLSQP polish with adaptive beta."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
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
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Gradient of the smooth maximum objective (log-sum-exp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max

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
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)

    # Final tiny correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)

    return h


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
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
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


def _projected_gradient_step(h, grad, lr):
    """
    One projected gradient step: move in -grad direction,
    then project onto {h: 0 <= h_i <= 1, sum(h) = N/2}.
    """
    h_new = h - lr * grad
    h_new = np.clip(h_new, 0.0, 1.0)
    h_new = _fix_sum(h_new)
    return h_new


def _nesterov_smooth_max(h0, beta, max_iter, lr_init=0.01):
    """
    Nesterov accelerated projected gradient on smooth max with box+sum constraints.
    Returns (h, score).
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)

    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init
    best_h = h.copy()
    best_score = _score(h)

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        val, grad = _smooth_max_val_grad(h, beta)

        # Gradient step
        h_new = _projected_gradient_step(h, grad, lr)

        # Nesterov extrapolation
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = np.clip(h_next, 0.0, 1.0)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        # Adaptive learning rate: decay slowly
        if it % 100 == 99:
            lr = max(0.002, lr * 0.9)

        # Track best
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    return best_h, best_score


def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """
    Run SLSQP optimization on the smooth max objective.
    """
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
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _optimize_spectral(h_init, K, beta, maxiter, lr_init=0.01):
    """
    Optimize in spectral coefficient space using L-BFGS-B with penalty for bounds.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)

    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h) ** 2) + np.sum(np.maximum(0, h - 1) ** 2)
        val, _ = _smooth_max_val_grad(h, beta)
        return val + 1.0 * penalty

    def grad(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        _, grad_h = _smooth_max_val_grad(h, beta)

        # Gradient of penalty
        penalty_grad = np.zeros(N)
        penalty_grad[h < 0] += -2.0 * h[h < 0]
        penalty_grad[h > 1] += 2.0 * (h[h > 1] - 1.0)

        total_grad_h = grad_h + penalty_grad

        # Chain rule: d(total)/d(coeffs)
        n_modes = len(coeffs) // 2
        i = np.arange(N)
        grad_coeffs = np.zeros(len(coeffs))
        for k in range(1, n_modes + 1):
            cos_k = np.cos(2.0 * np.pi * k * i / N)
            sin_k = np.sin(2.0 * np.pi * k * i / N)
            grad_coeffs[2 * (k - 1)] = np.sum(total_grad_h * cos_k)
            grad_coeffs[2 * (k - 1) + 1] = np.sum(total_grad_h * sin_k)

        return grad_coeffs

    # Use L-BFGS-B for spectral optimization
    try:
        res = minimize(obj, coeffs0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < obj(coeffs0):
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass

    # Fallback: Nesterov in spectral space
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    coeffs_prev = coeffs.copy()
    t_k = 1.0
    lr = lr_init
    best_coeffs = coeffs.copy()
    best_val = obj(coeffs)

    for it in range(max(100, maxiter)):
        if time.time() > _DEADLINE - 10:
            break
        val = obj(coeffs)
        g = grad(coeffs)

        coeffs_new = coeffs - lr * g
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        coeffs_next = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs_prev)

        coeffs_prev = coeffs
        coeffs = coeffs_next
        t_k = t_new

        if it % 50 == 49:
            lr = max(0.001, lr * 0.95)

        v = obj(coeffs)
        if v < best_val:
            best_val = v
            best_coeffs = coeffs.copy()

    h_opt = _spectral_h_from_coeffs(best_coeffs)
    return h_opt


def _final_polish(h0):
    """
    Multi-stage final polish with increasing beta for sharper max approximation.
    """
    h = h0.copy()
    
    # Stage 1: beta=100, moderate iterations
    if time.time() < _DEADLINE - 25:
        h = _run_slsqp(h, beta=100.0, maxiter=300, ftol=1e-12, gtol=1e-10)
        h = _fix_sum(h)
    
    # Stage 2: beta=300, more iterations
    if time.time() < _DEADLINE - 15:
        h = _run_slsqp(h, beta=300.0, maxiter=400, ftol=1e-13, gtol=1e-11)
        h = _fix_sum(h)
    
    # Stage 3: beta=500, tightest tolerance
    if time.time() < _DEADLINE - 5:
        h = _run_slsqp(h, beta=500.0, maxiter=500, ftol=1e-14, gtol=1e-12)
        h = _fix_sum(h)
    
    return h


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

    i = np.arange(N)

    # --- Phase 1: Diverse structured starts ---
    starts = []

    # Cosine fundamental
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_cos))

    # Sine fundamental
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_sin))

    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))

    # Three-frequency
    h_3f = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N) + 0.15 * np.cos(6.0 * np.pi * i / N)
    starts.append(_fix_sum(h_3f))

    # Four-frequency
    h_4f = 0.5 + 0.2 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N) + 0.1 * np.cos(6.0 * np.pi * i / N) + 0.08 * np.cos(8.0 * np.pi * i / N)
    starts.append(_fix_sum(h_4f))

    # Ramp pattern
    h_ramp = np.linspace(0.2, 0.8, N)
    starts.append(_fix_sum(h_ramp))

    # Triangular wave
    h_tri = 0.5 + 0.3 * (2 * np.abs(2 * (i / N - 0.5) - 1) - 1)
    starts.append(_fix_sum(h_tri))

    # Alternating pattern
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    starts.append(_fix_sum(h_alt))

    # Block pattern
    h_block = np.zeros(N)
    h_block[:N//2] = 0.8
    h_block[N//2:] = 0.2
    starts.append(_fix_sum(h_block))

    # Random sines
    for _ in range(15):
        f = np.random.uniform(1, 8)
        phase = np.random.uniform(0, 2 * np.pi)
        h_s = 0.5 + 0.5 * np.sin(2 * np.pi * f * i / N + phase)
        starts.append(_fix_sum(h_s))

    # Random
    for _ in range(15):
        h_r = np.random.rand(N)
        starts.append(_fix_sum(h_r))

    for h0 in starts:
        _update_best(h0)

    # --- Phase 2: Spectral optimization from best start ---
    for K in [2, 4, 8, 16, 32]:
        if time.time() - t0 > _DEADLINE - 40:
            break
        h_spec = _optimize_spectral(best_h, K, beta=50.0, maxiter=200)
        _update_best(h_spec)

    # --- Phase 3: Nesterov accelerated gradient refinement ---
    if best_h is not None and time.time() < _DEADLINE - 30:
        h_cand, s_cand = _nesterov_smooth_max(best_h, beta=50.0, max_iter=500, lr_init=0.01)
        _update_best(h_cand)

    if best_h is not None and time.time() < _DEADLINE - 20:
        h_cand2, s_cand2 = _nesterov_smooth_max(best_h, beta=100.0, max_iter=400, lr_init=0.005)
        _update_best(h_cand2)

    # --- Phase 4: Random restarts with spectral polish ---
    if time.time() < _DEADLINE - 35:
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 25:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            h_spec_rand = _optimize_spectral(h_rand, K=8, beta=50.0, maxiter=100)
            _update_best(h_spec_rand)

    # --- Phase 5: Final multi-beta polish ---
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_polished = _final_polish(best_h)
        _update_best(h_polished)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
