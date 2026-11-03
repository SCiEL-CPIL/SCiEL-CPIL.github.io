"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated overlap computation, exact Euclidean projection,
Nesterov accelerated projected gradient on log-sum-exp smooth surrogate,
and scipy SLSQP polish. Multi-phase beta schedule for coarse-to-fine optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 142  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET

    # Quick check if already feasible
    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)

    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0

    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))

    # Ensure bracket
    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    # Bisection for tau
    for _ in range(60):
        if time.time() > _DEADLINE - 2:
            break
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if s > target:
            lo = mid
        else:
            hi = mid

    tau = (lo + hi) / 2.0
    h = np.clip(h - tau, 0.0, 1.0)
    return h


def _smooth_max_val_grad(h, beta):
    """Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
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


def _nesterov_smooth_max(h0, beta, max_iter=500):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h)

    h_prev = h.copy()
    t_k = 1.0
    lr = 0.01

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(h, beta)

        # Gradient step
        h_new = h - lr * grad

        # Project onto feasible set
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


def _slsqp_polish(h0, beta, maxiter, ftol):
    """SLSQP optimization on smooth max surrogate."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - _TARGET,
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
        if res.success or res.fun < float('inf'):
            h_res = _project(res.x)
            return h_res, _score(h_res)
    except Exception:
        pass
    return h0, _score(h0)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

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

    # Diverse structured starts
    starts = []

    # Pure random
    for _ in range(10):
        h_rand = np.random.rand(N)
        starts.append(_project(h_rand))

    # Random sine waves
    for _ in range(10):
        f = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2 * np.pi)
        h_sine = 0.5 + 0.5 * np.sin(2 * np.pi * f * i / N + phase)
        starts.append(_project(h_sine))

    # Cosine fundamental
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_project(h_cos))

    # Sine fundamental
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_project(h_sin))

    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_project(h_2f))

    # Three-frequency
    h_3f = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N) + 0.1 * np.cos(6.0 * np.pi * i / N)
    starts.append(_project(h_3f))

    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_project(h_saw))

    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)

    # Phase 2: Nesterov accelerated gradient refinement
    if best_h is not None and time.time() < _DEADLINE - 15:
        # First pass with moderate beta
        h_candidate, s_candidate = _nesterov_smooth_max(best_h, beta=50.0, max_iter=400)
        if s_candidate < best_score:
            _update_best(h_candidate)

        # Second pass with higher beta
        if time.time() < _DEADLINE - 15:
            h_candidate2, s_candidate2 = _nesterov_smooth_max(best_h, beta=100.0, max_iter=400)
            if s_candidate2 < best_score:
                _update_best(h_candidate2)

        # Try from a perturbed start
        if time.time() < _DEADLINE - 15:
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = _project(h_pert)
            h_candidate3, s_candidate3 = _nesterov_smooth_max(h_pert, beta=80.0, max_iter=300)
            if s_candidate3 < best_score:
                _update_best(h_candidate3)

        # Another perturbation with different scale
        if time.time() < _DEADLINE - 15:
            h_pert2 = best_h + 0.1 * np.random.randn(N)
            h_pert2 = _project(h_pert2)
            h_candidate4, s_candidate4 = _nesterov_smooth_max(h_pert2, beta=60.0, max_iter=300)
            if s_candidate4 < best_score:
                _update_best(h_candidate4)

    # Phase 3: SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_res, s_res = _slsqp_polish(best_h, beta=100.0, maxiter=200, ftol=1e-10)
        if s_res < best_score:
            _update_best(h_res)

        # Final high-precision polish
        if time.time() < _DEADLINE - 5:
            h_res2, s_res2 = _slsqp_polish(best_h, beta=200.0, maxiter=300, ftol=1e-12)
            if s_res2 < best_score:
                _update_best(h_res2)

    # Phase 4: Final Nesterov with very high beta
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_final, s_final = _nesterov_smooth_max(best_h, beta=300.0, max_iter=500)
        if s_final < best_score:
            _update_best(h_final)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h)
    return best_h
