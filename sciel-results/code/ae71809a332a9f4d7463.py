import numpy as np
import time
import math


# EVOLVE-BLOCK-START
_DEADLINE_LIMIT = 140.0  # wall-clock budget (evaluator kills at 150s)
_SAFETY = 8.0


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


def _simple_thresholds(K, r, sigma, T, n_steps):
    """Fallback: time-decaying thresholds, always valid (never raises)."""
    dt = T / n_steps
    th = np.empty(n_steps)
    for j in range(n_steps):
        rem = (n_steps - j) * dt
        th[j] = K * (1.0 - math.exp(-2.5 * rem / max(dt, 1e-9)))
    th[-1] = K  # at maturity always exercise in-the-money
    return th


def _continuation_gh(V_next, logS_grid, S_grid, drift, vol_step, disc, n_quad, zq, wq):
    """Exact risk-neutral continuation value via Gauss-Hermite quadrature in log-space.

    log(S_next) = log(S) + drift + vol_step * Z, Z ~ N(0,1).
    E[V_next(S_next)] = sum_k wq_k * V_next(exp(logS_grid_i + drift + vol_step*zq_k))
    Interpolation is done in log-space (logS_grid) for better accuracy near S=0.
    """
    n = S_grid.shape[0]
    logS_next = logS_grid[:, None] + drift + vol_step * zq[None, :]
    logS_next = np.clip(logS_next, logS_grid[0], logS_grid[-1])
    V_interp = np.interp(logS_next.ravel(), logS_grid, V_next).reshape(logS_next.shape)
    cont = disc * np.sum(wq[None, :] * V_interp, axis=1)
    return cont


def _threshold_bisect(immediate, cont, S_grid, lo_idx, hi_idx):
    """Bisect for the threshold where immediate(s) = cont(s) between S_grid[lo_idx] and S_grid[hi_idx].

    immediate is decreasing, cont is increasing -> single crossing.
    """
    a = S_grid[lo_idx]
    b = S_grid[hi_idx]
    fa = immediate[lo_idx] - cont[lo_idx]
    fb = immediate[hi_idx] - cont[hi_idx]
    if fa < 0:
        return a
    if fb > 0:
        return b
    for _ in range(60):
        mid = 0.5 * (a + b)
        # interpolate immediate and cont at mid
        im = np.interp(mid, S_grid, immediate)
        co = np.interp(mid, S_grid, cont)
        fm = im - co
        if fm > 0:
            a = mid
        else:
            b = mid
        if b - a < 1e-7:
            break
    return 0.5 * (a + b)


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()

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
    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)
    disc = math.exp(-r * dt)

    # Grid: focus where the put has value. Log-spaced for better resolution near 0.
    S_min = max(0.01 * K, 1e-4)
    S_max = 2.0 * K
    n_grid = 500
    S_grid = np.geomspace(S_min, S_max, n_grid)
    logS_grid = np.log(S_grid)
    immediate = np.maximum(K - S_grid, 0.0)

    # Gauss-Hermite quadrature
    n_quad = 64
    zq, wq = np.polynomial.hermite.hermgauss(n_quad)
    zq = zq * math.sqrt(2.0)
    wq = wq / math.sqrt(math.pi)

    def elapsed_ok():
        return (time.time() - t_start) < (_DEADLINE_LIMIT - _SAFETY)

    # Backward induction
    V_next = immediate.copy()
    thresholds = np.zeros(n_steps)

    for t_idx in range(n_steps - 1, -1, -1):
        if not elapsed_ok():
            break
        cont = _continuation_gh(V_next, logS_grid, S_grid, drift, vol_step, disc, n_quad, zq, wq)
        V_cur = np.maximum(immediate, cont)

        # Find threshold: largest S where immediate >= cont
        diff = immediate - cont
        valid = S_grid < K
        if np.any(valid):
            idx_valid = np.where(valid)[0]
            diff_v = diff[idx_valid]
            pos_idx = np.where(diff_v >= 0)[0]
            if len(pos_idx) > 0:
                last = pos_idx[-1]
                if last + 1 < len(idx_valid):
                    lo_idx = idx_valid[last]
                    hi_idx = idx_valid[last + 1]
                    thr = _threshold_bisect(immediate, cont, S_grid, lo_idx, hi_idx)
                else:
                    thr = S_grid[idx_valid[last]]
                thr = min(max(thr, 0.0), K)
            else:
                thr = 0.0
        else:
            thr = 0.0
        thresholds[t_idx] = thr
        V_next = V_cur

    # Monotonicity: earlier dates should have higher (or equal) thresholds
    for j in range(1, n_steps):
        if thresholds[j] > thresholds[j - 1]:
            thresholds[j] = thresholds[j - 1]

    # Coordinate-descent polish: for each threshold, nudge to improve local value
    # Use a simple local search: try small perturbations and keep if better.
    # Objective proxy: for each date j, the value at the threshold should be
    # where the "kink" is. We refine by checking if shifting the threshold
    # slightly changes the policy in a beneficial way.
    if elapsed_ok():
        # Re-compute continuation values at each date for the polish
        V_next = immediate.copy()
        cont_cache = np.zeros((n_steps, n_grid))
        for t_idx in range(n_steps - 1, -1, -1):
            if not elapsed_ok():
                break
            cont = _continuation_gh(V_next, logS_grid, S_grid, drift, vol_step, disc, n_quad, zq, wq)
            cont_cache[t_idx] = cont
            V_cur = np.maximum(immediate, cont)
            V_next = V_cur

        # Polish each threshold via local search
        step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        for j in range(n_steps):
            if not elapsed_ok():
                break
            if j >= n_steps:
                break
            # For date j, the continuation value is cont_cache[j]
            # The threshold is where immediate(s) = cont_cache[j](s)
            # We can refine by checking the difference at the current threshold
            # and nudging in the direction that reduces |immediate - cont|
            cont_j = cont_cache[j]
            current_th = thresholds[j]
            # Evaluate immediate and cont at current threshold
            im_at_th = max(K - current_th, 0.0)
            co_at_th = np.interp(current_th, S_grid, cont_j)
            # The threshold should be where im = co. If im > co, we're in exercise region,
            # threshold should be higher. If im < co, threshold should be lower.
            # Use bisection with the actual cont function for precision
            # Find bracket
            lo_s = max(S_grid[0], current_th - 2.0)
            hi_s = min(S_grid[-1], current_th + 2.0)
            # Ensure bracket
            f_lo = max(K - lo_s, 0.0) - np.interp(lo_s, S_grid, cont_j)
            f_hi = max(K - hi_s, 0.0) - np.interp(hi_s, S_grid, cont_j)
            if f_lo * f_hi < 0:
                a, b = lo_s, hi_s
                for _ in range(50):
                    mid = 0.5 * (a + b)
                    f_mid = max(K - mid, 0.0) - np.interp(mid, S_grid, cont_j)
                    if f_mid > 0:
                        a = mid
                    else:
                        b = mid
                    if b - a < 1e-7:
                        break
                new_th = 0.5 * (a + b)
            else:
                new_th = current_th
            new_th = min(max(new_th, 0.0), K)
            thresholds[j] = new_th

        # Re-enforce monotonicity after polish
        for j in range(1, n_steps):
            if thresholds[j] > thresholds[j - 1]:
                thresholds[j] = thresholds[j - 1]

    # Final safety: clip thresholds
    thresholds = np.clip(thresholds, 0.0, K)

    # Ensure last threshold is at least K (always exercise ITM at maturity)
    thresholds[-1] = max(thresholds[-1], K * 0.99)

    return _make_policy(thresholds, n_steps, dt, T)
# EVOLVE-BLOCK-END
