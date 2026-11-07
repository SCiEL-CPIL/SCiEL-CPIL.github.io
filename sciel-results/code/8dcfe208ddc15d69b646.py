import numpy as np
from scipy.optimize import minimize
import time

N = 50
m = 2 * N - 1  # 99
m_plus_1 = m + 1  # 100
w = 1.0 / m_plus_1  # spacing


def _evaluate(h):
    """Compute the objective value for a given set of heights h."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0

    g = np.convolve(h, h)
    y = np.zeros(m + 2)
    y[1:-1] = g

    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])

    norm_1 = np.sum(np.abs(g)) / m_plus_1
    norm_inf = np.max(np.abs(g))

    if norm_1 * norm_inf < 1e-300:
        return 0.0

    return l2 / (norm_1 * norm_inf)


def _log_sum_exp(x, beta=1.0):
    """Differentiable approximation to max(x) using log-sum-exp."""
    scaled = beta * x
    mx = np.max(scaled)
    if mx > 700:
        return np.max(x)
    return (1.0 / beta) * (mx + np.log(np.sum(np.exp(scaled - mx))))


def _negative_value_smooth(h, beta=100.0):
    """Smooth negative objective (log-sum-exp for norm_inf)."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or not np.all(np.isfinite(h)):
        return 1e10

    g = np.convolve(h, h)
    y = np.zeros(m + 2)
    y[1:-1] = g

    y_sq = y ** 2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])

    norm_1 = np.sum(np.abs(g)) / m_plus_1
    abs_g = np.abs(g)
    norm_inf_smooth = _log_sum_exp(abs_g, beta=beta)

    if norm_1 * norm_inf_smooth < 1e-300:
        return 1e10

    return -l2 / (norm_1 * norm_inf_smooth)


def _negative_value_grad(h, beta=100.0):
    """Negative objective with numerical gradient."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N or not np.all(np.isfinite(h)):
        return 1e10, np.zeros(N)

    val = _negative_value_smooth(h, beta=beta)
    eps = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        h_p = h.copy()
        h_p[i] += eps
        h_m = h.copy()
        h_m[i] = max(h[i] - eps, 0.0)
        grad[i] = (_negative_value_smooth(h_p, beta=beta) -
                   _negative_value_smooth(h_m, beta=beta)) / (h_p[i] - h_m[i])
    return val, grad


def _parametric_h(params, kind='gaussian'):
    """Generate h from parametric form."""
    if kind == 'gaussian':
        sigma, center, amp = params
        x = np.linspace(-0.5, 0.5, N)
        h = amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    elif kind == 'super_gaussian':
        sigma, center, amp, n = params
        x = np.linspace(-0.5, 0.5, N)
        h = amp * np.exp(-((x - center) / sigma) ** (2 * n))
    elif kind == 'polynomial':
        a0, a1, a2, a3 = params
        x = np.linspace(-0.5, 0.5, N)
        h = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3
        h = np.maximum(h, 0)
    elif kind == 'flat_top':
        center, width, amp = params
        x = np.linspace(-0.5, 0.5, N)
        h = amp * np.exp(-((x - center) / width) ** 4)
    elif kind == 'hamming':
        center, width, amp = params
        x = np.linspace(-0.5, 0.5, N)
        xn = (x - center) / width
        h = amp * (0.54 + 0.46 * np.cos(np.pi * xn))
        h = np.where(np.abs(xn) <= 1.0, h, 0.0)
    elif kind == 'blackman':
        center, width, amp = params
        x = np.linspace(-0.5, 0.5, N)
        xn = (x - center) / width
        h = amp * (0.42 + 0.5 * np.cos(np.pi * xn) + 0.08 * np.cos(2 * np.pi * xn))
        h = np.where(np.abs(xn) <= 1.0, h, 0.0)
    elif kind == 'two_gauss':
        c1, c2, s1, s2, a1, a2 = params
        x = np.linspace(-0.5, 0.5, N)
        h = a1 * np.exp(-0.5 * ((x - c1) / s1) ** 2) + a2 * np.exp(-0.5 * ((x - c2) / s2) ** 2)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return np.maximum(h, 0.0)


def _optimize_parametric(kind, start_time, deadline):
    """Optimize a parametric family."""
    best_val = 0.0
    best_h = None

    if kind == 'gaussian':
        bounds = [(0.01, 0.5), (-0.4, 0.4), (0.1, 5.0)]
        x0 = [0.15, 0.0, 1.0]
    elif kind == 'super_gaussian':
        bounds = [(0.01, 0.5), (-0.4, 0.4), (0.1, 5.0), (0.5, 5.0)]
        x0 = [0.15, 0.0, 1.0, 1.5]
    elif kind == 'polynomial':
        bounds = [(0.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
        x0 = [1.0, 0.0, 2.0, 0.0]
    elif kind == 'flat_top':
        bounds = [(-0.4, 0.4), (0.01, 0.5), (0.1, 5.0)]
        x0 = [0.0, 0.15, 1.0]
    elif kind == 'hamming':
        bounds = [(-0.4, 0.4), (0.01, 0.5), (0.1, 5.0)]
        x0 = [0.0, 0.2, 1.0]
    elif kind == 'blackman':
        bounds = [(-0.4, 0.4), (0.01, 0.5), (0.1, 5.0)]
        x0 = [0.0, 0.2, 1.0]
    elif kind == 'two_gauss':
        bounds = [(-0.4, 0.4), (-0.4, 0.4), (0.01, 0.3), (0.01, 0.3),
                  (0.1, 5.0), (0.1, 5.0)]
        x0 = [-0.15, 0.15, 0.1, 0.1, 1.0, 1.0]
    else:
        return 0.0, None

    # Try multiple starting points
    starts = [x0]
    # Add perturbed starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed = [p * (1 + 0.3 * rng.randn()) for p in x0]
        starts.append(perturbed)

    for x0_trial in starts:
        if time.time() - start_time > deadline - 10:
            break
        try:
            res = minimize(
                fun=lambda p: _negative_value_smooth(_parametric_h(p, kind), beta=100.0),
                x0=x0_trial,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if np.all(np.isfinite(res.x)):
                h_trial = _parametric_h(res.x, kind)
                val = _evaluate(h_trial)
                if val > best_val:
                    best_val = val
                    best_h = h_trial.copy()
        except Exception:
            pass

    return best_val, best_h


def construct():
    """Construct 50 nonnegative step heights to maximize the objective."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Start with uniform seed
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = _evaluate(h0)

    # 1. Parametric family optimizations
    parametric_kinds = ['gaussian', 'super_gaussian', 'polynomial',
                        'flat_top', 'hamming', 'blackman', 'two_gauss']
    for kind in parametric_kinds:
        if time.time() - start_time > deadline - 30:
            break
        pval, ph = _optimize_parametric(kind, start_time, deadline)
        if ph is not None and pval > best_val:
            best_val = pval
            best_h = ph.copy()

    # 2. Full 50-dim L-BFGS-B from best parametric result
    if time.time() - start_time < deadline - 20:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() - start_time > deadline - 10:
                break
            try:
                result = minimize(
                    fun=lambda x: _negative_value_smooth(x, beta=beta),
                    x0=best_h,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                if np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                    val = _evaluate(result.x)
                    if val > best_val:
                        best_val = val
                        best_h = result.x.copy()
            except Exception:
                pass

    # 3. Final refinement with highest beta
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                fun=lambda x: _negative_value_smooth(x, beta=500.0),
                x0=best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 30}
            )
            if np.all(result.x >= 0) and np.all(np.isfinite(result.x)):
                val = _evaluate(result.x)
                if val > best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass

    # Ensure non-negativity and finiteness
    best_h = np.maximum(best_h, 0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)

    return list(best_h)
