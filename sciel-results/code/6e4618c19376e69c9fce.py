import numpy as np
import time
from math import log, sqrt, exp, erf, pi


# EVOLVE-BLOCK-START
def _norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _european_put(S, K, r, sigma, tau):
    """Exact European put price under GBM."""
    if tau <= 0:
        return np.maximum(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    return K * exp(-r * tau) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _compute_cont(V_next, grid, expx, x_weights, n_grid):
    """Vectorized continuation via Gauss-Hermite quadrature."""
    S_next = grid[:, None] * expx[None, :]
    S_next_clipped = np.clip(S_next, grid[0], grid[-1])
    idx = np.searchsorted(grid, S_next_clipped)
    idx = np.clip(idx, 1, n_grid - 1)
    x0 = grid[idx - 1]
    x1 = grid[idx]
    f0 = V_next[idx - 1]
    f1 = V_next[idx]
    w = (S_next_clipped - x0) / (x1 - x0 + 1e-300)
    V_interp = f0 * (1.0 - w) + f1 * w
    return V_interp @ x_weights


def _simple_thresholds(K, r, sigma, T, n_steps):
    """Fallback: crude time-decaying thresholds, always valid (never raises)."""
    dt = T / n_steps
    th = np.empty(n_steps)
    for j in range(n_steps):
        # crude: exercise when S < K * (1 - small) early, tighter later
        frac = (j + 1) / n_steps
        th[j] = K * (0.98 - 0.05 * frac)
    return th


def _make_policy(thresholds, n_steps, dt, T):
    class ExercisePolicy:
        def __init__(self, thresholds, n_steps, dt, T):
            self.thresholds = np.asarray(thresholds, dtype=np.float64)
            self.n_steps = n_steps
            self.dt = dt
            self.T = T

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            if time is None:
                t = 0.0
            else:
                try:
                    t = float(time)
                except Exception:
                    t = 0.0
            if self.dt > 0:
                j = int(round(t / self.dt))
            else:
                j = 1
            j = max(1, min(j, self.n_steps))
            threshold = self.thresholds[j - 1]
            return cs <= threshold

    return ExercisePolicy(thresholds, n_steps, dt, T)


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    DEADLINE = 150.0
    SAFETY = 6.0

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12

    def _get(obj, *names, default=None):
        for nm in names:
            if isinstance(obj, dict):
                if nm in obj:
                    return obj[nm]
            elif hasattr(obj, nm):
                return getattr(obj, nm)
        return default

    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', default=T))
        n_steps = int(_get(contract, 'n_dates', 'num_dates', 'exercise_dates', default=n_steps))
    except Exception:
        pass

    if n_steps < 1:
        n_steps = 1
    dt = T / n_steps if n_steps > 0 else T
    mu_log = (r - 0.5 * sigma * sigma) * dt
    vol_log = sigma * np.sqrt(dt)

    # Grid: focus on region where put has value
    S_lo = max(1e-3, K * 0.05)
    S_hi = K * 2.0
    n_grid = 800
    grid = np.linspace(S_lo, S_hi, n_grid)
    payoff = np.maximum(K - grid, 0.0)

    def elapsed_ok():
        return (time.time() - t_start) < (DEADLINE - SAFETY)

    thresholds = None
    try:
        from numpy.polynomial.hermite import hermgauss
        z_nodes, z_weights = hermgauss(32)
        x_nodes = mu_log + vol_log * sqrt(2.0) * z_nodes
        x_weights = z_weights / sqrt(pi)
        expx = exp(x_nodes)

        V = np.empty((n_steps + 1, n_grid))
        V[n_steps, :] = payoff

        for j in range(n_steps - 1, -1, -1):
            if not elapsed_ok():
                break
            cont = _compute_cont(V[j + 1, :], grid, expx, x_weights, n_grid)
            V[j, :] = np.maximum(payoff, cont)

        # Compute thresholds via bisection
        th = np.empty(n_steps)
        for j in range(n_steps):
            if not elapsed_ok():
                break
            cont = _compute_cont(V[j + 1, :], grid, expx, x_weights, n_grid)
            ex_mask = payoff >= cont - 1e-12
            if not np.any(ex_mask):
                th[j] = grid[0]
            else:
                idx_ex = np.where(ex_mask)[0]
                hi = idx_ex[-1]
                while hi + 1 < n_grid and payoff[hi + 1] >= cont[hi + 1] - 1e-12:
                    hi += 1
                if hi >= n_grid - 1:
                    th[j] = grid[hi]
                else:
                    a = grid[hi]
                    b = grid[hi + 1]

                    def f_thresh(s, j=j):
                        s_next = s * expx
                        s_next_c = np.clip(s_next, grid[0], grid[-1])
                        idxx = np.searchsorted(grid, s_next_c)
                        idxx = np.clip(idxx, 1, n_grid - 1)
                        x0_ = grid[idxx - 1]
                        x1_ = grid[idxx]
                        f0_ = V[j + 1, idxx - 1]
                        f1_ = V[j + 1, idxx]
                        w_ = (s_next_c - x0_) / (x1_ - x0_ + 1e-300)
                        Vi_ = f0_ * (1.0 - w_) + f1_ * w_
                        c_ = Vi_ @ x_weights
                        return max(K - s, 0.0) - c_

                    fa = f_thresh(a)
                    fb = f_thresh(b)
                    if fa < 0:
                        th[j] = a
                    elif fb > 0:
                        th[j] = b
                    else:
                        for _ in range(50):
                            mid = 0.5 * (a + b)
                            fm = f_thresh(mid)
                            if fm > 0:
                                a = mid
                            else:
                                b = mid
                            if b - a < 1e-5:
                                break
                        th[j] = 0.5 * (a + b)
        if elapsed_ok():
            thresholds = th
    except Exception:
        thresholds = None

    if thresholds is None or not np.all(np.isfinite(thresholds)):
        thresholds = _simple_thresholds(K, r, sigma, T, n_steps)

    # Enforce monotonic non-increasing across dates (earlier dates: higher threshold).
    for j in range(1, n_steps):
        if thresholds[j] > thresholds[j - 1]:
            thresholds[j] = thresholds[j - 1]

    # Keep thresholds within a sane range.
    thresholds = np.clip(thresholds, 0.0, K * 1.5)

    return _make_policy(thresholds, n_steps, dt, T)
# EVOLVE-BLOCK-END
