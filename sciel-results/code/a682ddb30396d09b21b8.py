"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: exploit Toeplitz structure for O(n log n) evaluation via FFT,
use accelerated projected gradient (FISTA) on smooth-max surrogate for the main
optimization loop, with SLSQP only for final polish. Bisection on C_max uses
feasibility via the same accelerated gradient.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
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
    Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp.
    Gradient computed analytically using the structure of C_k.
    """
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
        # C_k = 1 - (2/n)*r_k where r_k = sum_{i=k}^{n-1} h_i h_{i-k}
        # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k else 0 + h_{j+k} if j+k<n else 0)
        if k > 0:
            # h_{j-k} for j >= k: indices j=k..n-1, h[0..n-1-k]
            grad[k:] += factor * h[:n - k]
        # h_{j+k} for j+k < n: indices j=0..n-1-k, h[k..n-1]
        if k < n:
            grad[:n - k] += factor * h[k:]

    return val, grad


def _project(h, target_sum):
    """
    Project h onto {0<=h_i<=1, sum(h)=target_sum} using iterative water-filling.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    for _ in range(200):
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        diff = target_sum - np.sum(h)
    # Final tiny correction
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    return _project(h, N / 2.0)


def _fista_optimize(h0, beta, max_iter, ftol, target_sum=None):
    """
    Accelerated projected gradient (FISTA) for minimizing smooth max of C_k.
    Returns optimized h.
    """
    if target_sum is None:
        target_sum = N / 2.0

    n = N
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, target_sum)

    y = h.copy()
    z = h.copy()
    t = 1.0
    step_size = 1.0 / beta  # Lipschitz constant estimate; we'll use backtracking

    prev_val = float('inf')
    prev_grad = None

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        val, grad = _smooth_max_val_grad(y, beta)

        # Backtracking line search
        lr = step_size
        for _ in range(30):
            y_new = y - lr * grad
            y_new = _project(y_new, target_sum)
            val_new, _ = _smooth_max_val_grad(y_new, beta)
            if val_new <= val - 0.5 * lr * np.dot(grad, y_new - y):
                break
            lr *= 0.5

        # FISTA update
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = y_new
        z = y + ((t - 1.0) / t_new) * (y - h)
        h = y.copy()
        t = t_new

        if abs(prev_val - val) < ftol:
            break
        prev_val = val

    return h


def _slsqp_polish(h0, beta, maxiter, ftol):
    """
    Run SLSQP optimization with smooth max objective for final polish.
    """
    n = N

    def obj_func(x):
        val, _ = _smooth_max_val_grad(x, beta)
        return val

    def jac_func(x):
        _, grad = _smooth_max_val_grad(x, beta)
        return grad

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(n)}]
    bounds = [(0.0, 1.0)] * n

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        return res.x
    except Exception:
        return h0


def _feasibility_check(c_max, h0, maxiter=100):
    """
    Check if there exists h such that max(C_k) <= c_max, sum(h)=N/2, 0<=h_i<=1.
    Uses FISTA on a penalty objective.
    """
    n = N
    target_sum = n / 2.0

    def penalty_obj_grad(h):
        cs = _linear_overlap_values(h)
        violations = np.maximum(0.0, cs - c_max)
        val = 0.5 * np.sum(violations ** 2)
        grad = np.zeros(n)
        mask = cs > c_max
        if np.any(mask):
            for k in np.where(mask)[0]:
                w = 2.0 * (cs[k] - c_max)
                factor = -(2.0 / n) * w
                if k > 0:
                    grad[k:] += factor * h[:n - k]
                if k < n:
                    grad[:n - k] += factor * h[k:]
        return val, grad

    # Simple projected gradient for feasibility
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, target_sum)
    y = h.copy()
    z = h.copy()
    t = 1.0
    lr = 0.01

    for it in range(maxiter):
        if time.time() > _DEADLINE - 10:
            break
        val, grad = penalty_obj_grad(y)
        y_new = _project(y - lr * grad, target_sum)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = y_new
        z = y + ((t - 1.0) / t_new) * (y - h)
        h = y.copy()
        t = t_new

    h_sol = _project(h, target_sum)
    cs_sol = _linear_overlap_values(h_sol)
    return np.max(cs_sol) <= c_max + 1e-6, h_sol


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

    # 1. Analytical starts
    i = np.arange(N)

    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)

    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)

    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    h_alt = _fix_sum(h_alt)
    _update_best(h_alt)

    # 2. Bisection to bracket optimal C_max
    lo = 0.0
    hi = best_score

    max_bisection_iter = 12
    for it in range(max_bisection_iter):
        if time.time() - t0 > _DEADLINE - 60:
            break

        mid = (lo + hi) / 2.0
        feasible, h_sol = _feasibility_check(mid, best_h, maxiter=60)

        if feasible:
            hi = mid
            _update_best(h_sol)
        else:
            lo = mid

    # 3. FISTA optimization from best point with increasing beta
    if time.time() - t0 < _DEADLINE - 40:
        for beta in [30.0, 80.0, 200.0]:
            if time.time() - t0 > _DEADLINE - 20:
                break
            h_opt = _fista_optimize(best_h, beta, max_iter=300, ftol=1e-10)
            _update_best(h_opt)

    # 4. Try a few random starts if time permits
    if time.time() - t0 < _DEADLINE - 30:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            h_opt = _fista_optimize(h_rand, beta=50.0, max_iter=200, ftol=1e-9)
            _update_best(h_opt)

    # 5. Final SLSQP polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polished = _slsqp_polish(best_h, beta=100.0, maxiter=200, ftol=1e-12)
        _update_best(h_polished)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
