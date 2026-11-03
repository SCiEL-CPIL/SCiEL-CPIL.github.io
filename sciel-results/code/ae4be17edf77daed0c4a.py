"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines SLSQP (from current) with Nesterov accelerated projected gradient (from inspiration 1).
Uses FFT-accelerated overlap, log-sum-exp smooth-max surrogate, multiple initializations,
and a final high-precision polish. The accelerated gradient method provides a fast global
search complement to SLSQP's local refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 138  # Safety margin before 150s kill


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


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Smooth maximum of the linear overlap values using log-sum-exp.
    Returns (value, gradient) vectorized via FFT-based gradient.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max

    val = m + np.log(sum_exp) / beta

    # Gradient computation
    grad = np.zeros(n)
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
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
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)

    return h


def _project(h):
    """Project h onto {0 <= h_i <= 1, sum(h) = N/2}."""
    h = np.clip(h, 0.0, 1.0)
    return _fix_sum(h)


def _run_slsqp(h0, beta, maxiter, ftol):
    """Run SLSQP optimization with smooth max objective."""
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
        return res.x
    except Exception:
        return h0


def _accelerated_gradient(h0, beta, max_iter=300, lr_init=0.01):
    """
    Nesterov accelerated projected gradient on smooth max surrogate.
    Projects onto box [0,1]^N and sum constraint.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)

        # Gradient step
        h_new = h - lr * grad

        # Project onto feasible set
        h_new = _project(h_new)

        # Nesterov momentum update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _project(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        # Adaptive learning rate decay
        if it % 50 == 49:
            lr = max(0.002, lr * 0.95)

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

    # Diverse initial candidates
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)

    # Constant wave
    h_const = np.full(N, 0.5)
    _update_best(h_const)

    # Sine-based
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)

    # Two-tone
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    _update_best(h_alt)

    # Two-frequency cosine
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_2f)

    # Random starts
    for _ in range(4):
        if time.time() - t0 > _DEADLINE - 30:
            break
        h_rand = np.random.uniform(0.05, 0.95, N)
        h_rand = _fix_sum(h_rand)
        _update_best(h_rand)

    def _try_method(h0, method, **kwargs):
        nonlocal best_h, best_score
        if time.time() - t0 > _DEADLINE - 10:
            return
        if method == 'slsqp':
            h_opt = _run_slsqp(h0, **kwargs)
        elif method == 'accel':
            h_opt = _accelerated_gradient(h0, **kwargs)
        else:
            return
        _update_best(h_opt)

    # Phase 1: Accelerated gradient from best candidates (fast global search)
    _try_method(best_h, 'accel', beta=50.0, max_iter=200, lr_init=0.01)

    # Phase 2: SLSQP refinement from accelerated result
    _try_method(best_h, 'slsqp', beta=30.0, maxiter=300, ftol=1e-9)

    # Phase 3: More accelerated gradient with higher beta
    _try_method(best_h, 'accel', beta=100.0, max_iter=300, lr_init=0.005)

    # Phase 4: SLSQP with higher beta for finer tracking of max
    _try_method(best_h, 'slsqp', beta=80.0, maxiter=400, ftol=1e-11)

    # Phase 5: Try from perturbed best
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_pert = best_h + 0.03 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            _try_method(h_pert, 'accel', beta=80.0, max_iter=200, lr_init=0.008)
            _try_method(best_h, 'slsqp', beta=80.0, maxiter=200, ftol=1e-10)

    # Phase 6: Final polish with very high beta SLSQP
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _try_method(best_h, 'slsqp', beta=200.0, maxiter=300, ftol=1e-13)

    # Phase 7: Final accelerated gradient polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        _try_method(best_h, 'accel', beta=200.0, max_iter=200, lr_init=0.003)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
