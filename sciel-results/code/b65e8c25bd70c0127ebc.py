"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Structural change: combine the direct full-vector smooth optimization (current)
with analytic gradients (inspiration) and a structured 1D transition-width
profile that exploits the rotational invariance of the overlap metric.

The 1D profile approach:
  - Assume h is a smooth "S-curve" (sigmoid) of width w, centered at some
    position, with a linear ramp to ensure sum = N/2.
  - The overlap max is a function of (w, center, amplitude) - a small search.
  - This gives a strong initial guess and a lower bound on the optimum.

Then polish with L-BFGS-B using analytic gradients of the smooth max.
"""
import time
import math
import numpy as np

N = 100
TARGET_SUM = N / 2.0
DEADLINE = time.time() + 135.0  # safety margin before the 150s kill


def _score(h):
    """Raw max overlap of h with shifted (1-h); h zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target, preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float)
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                diff3 = target - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """Compute soft max of x and its gradient using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum == 0:
        return max_x, np.zeros_like(x)
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _objective_and_grad(h, beta):
    """
    Compute the smooth objective value and its analytic gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    n = len(h)
    h = np.clip(h, 0.0, 1.0)

    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]

    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)

    grad_h = np.zeros(n)
    if n > 1:
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0

    penalty_weight = 100.0
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error

    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    return total_val, total_grad


def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """Optimize h using L-BFGS-B with analytic gradients and adaptive beta."""
    from scipy.optimize import minimize

    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)

    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]

    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break

        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val

        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad

        try:
            res = minimize(
                obj_func,
                h,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception:
            pass

    return current_score, h


def _local_search_polish(h, max_iters=100, deadline=None):
    """Deterministic local search to polish the solution."""
    h = h.copy()
    current_score = _score(h)

    if deadline is None:
        deadline = time.time() + 5.0

    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for delta in [0.005, -0.005, 0.01, -0.01]:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current_score, h


def _transition_profile(w, center, amp):
    """
    Construct a smooth S-curve profile:
      h_i = 0.5 + amp * tanh((i - center) / w)
    Then adjust to ensure sum = N/2 and values in [0,1].
    """
    i = np.arange(N, dtype=float)
    h = 0.5 + amp * np.tanh((i - center) / w)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def _opt_transition():
    """
    Optimize the 1D transition-width profile.
    Search over (w, center, amp) to minimize the max overlap.
    """
    best_score = float('inf')
    best_h = None

    # Grid search over width, center, amplitude
    for w in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]:
        for center in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]:
            for amp in [0.3, 0.4, 0.5, 0.6, 0.7]:
                h = _transition_profile(w, center, amp)
                s = _score(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()

    # Refine around the best with a finer local search
    if best_h is not None:
        from scipy.optimize import minimize

        def obj_params(params):
            h = _transition_profile(params[0], params[1], params[2])
            return _score(h)

        # Find the best grid point for the starting point
        best_params = None
        for w in [3.0, 4.0, 5.0, 7.0]:
            for center in [40.0, 50.0, 60.0]:
                for amp in [0.4, 0.5, 0.6]:
                    h = _transition_profile(w, center, amp)
                    s = _score(h)
                    if s < best_score or best_params is None:
                        best_params = [w, center, amp]
                        best_score = s

        if best_params is not None:
            res = minimize(
                obj_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12}
            )
            h = _transition_profile(res.x[0], res.x[1], res.x[2])
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h

    return best_score, best_h


def _sinusoidal_overlap(A, B):
    """Exact max overlap for h_i = 0.5 + A sin(2pi i/N) + B sin(2pi (i+1)/N)."""
    i = np.arange(N, dtype=float)
    h = 0.5 + A * np.sin(2.0 * math.pi * i / N) + B * np.sin(2.0 * math.pi * (i + 1) / N)
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap)), h


def _opt_sinusoidal():
    """Minimize the exact sinusoidal max overlap over (A, B)."""
    best = (float('inf'), None)
    grid = np.linspace(-0.45, 0.45, 41)
    for A in grid:
        for B in grid:
            s, h = _sinusoidal_overlap(A, B)
            if s < best[0]:
                best = (s, (A, B, h.copy()))
    A0, B0, _ = best[1]
    from scipy.optimize import minimize

    def obj(v):
        s, _ = _sinusoidal_overlap(v[0], v[1])
        return s
    res = minimize(obj, [A0, B0], method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12})
    s, h = _sinusoidal_overlap(res.x[0], res.x[1])
    if s < best[0]:
        best = (s, (res.x[0], res.x[1], h))
    return best


def construct():
    best_h = None
    best_score = float('inf')

    def consider(h):
        nonlocal best_h, best_score
        if h is None:
            return
        h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # ---- 1) Structured 1D transition profile (fast, strong lower bound) ----
    if time.time() < DEADLINE:
        s, h = _opt_transition()
        consider(h)

    # ---- 2) Sinusoidal ansatz (global, cheap) ----
    if time.time() < DEADLINE:
        s, (A, B, h) = _opt_sinusoidal()
        consider(h)

    # ---- 3) Full-vector L-BFGS-B with analytic gradients from multiple starts ----
    starts = []
    if best_h is not None:
        starts.append(best_h.copy())
    starts.append(np.full(N, 0.5))

    rng = np.random.default_rng(1234)
    for _ in range(3):
        A = rng.uniform(-0.4, 0.4)
        B = rng.uniform(-0.4, 0.4)
        i = np.arange(N, dtype=float)
        s0 = 0.5 + A * np.sin(2.0 * math.pi * i / N) + B * np.sin(2.0 * math.pi * (i + 1) / N)
        starts.append(np.clip(s0, 0.0, 1.0))

    # Add a few transition-profile starts
    for w, center, amp in [(3.0, 50.0, 0.5), (5.0, 45.0, 0.6), (4.0, 55.0, 0.4)]:
        h = _transition_profile(w, center, amp)
        starts.append(h)

    for x0 in starts:
        if time.time() > DEADLINE - 20.0:
            break
        try:
            s, h = _optimize_direct(x0, deadline=DEADLINE - 10.0)
            consider(h)
        except Exception:
            pass

    # ---- 4) Local search polish on the best ----
    if best_h is not None and time.time() < DEADLINE - 5.0:
        s, h = _local_search_polish(best_h, max_iters=80, deadline=DEADLINE - 2.0)
        consider(h)

    # ---- 5) Fallback ----
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N // 2] = A
        h[N // 2:] = B
        best_h = h

    # Final exact-sum guarantee
    diff = TARGET_SUM - best_h.sum()
    if abs(diff) > 1e-15:
        slack = np.where((best_h > 1e-9) & (best_h < 1.0 - 1e-9))[0]
        if len(slack):
            best_h[slack] += diff / len(slack)
            best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
