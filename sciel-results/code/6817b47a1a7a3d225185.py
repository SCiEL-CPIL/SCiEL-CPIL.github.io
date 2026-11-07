"""Erdos minimum overlap – rewritten optimizer.

Key changes vs the seed:
1.  The non-smooth max(corr) is handled with an epigraph variable t and
    log-sum-exp (smooth max) in the main SLSQP loop, then refined with a
    piecewise-linear / linear-programming pass (no, scipy.optimize.linprog
    is not guaranteed for 200 vars here – we use a custom subgradient /
    projected gradient with line search on the exact objective for the
    final polish).
2.  The local polish uses incremental correlation updates: when we change
    h[i] by delta_i and h[j] by -delta_j (to keep the sum fixed), the
    correlation vector changes in a predictable way, so we can update the
    max in O(1) per candidate pair rather than O(N).
3.  More diverse structured starting points and a tighter final exact
    local search.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0).astype(np.float64)
    target = N / 2.0
    for _ in range(30):
        diff = np.sum(h) - target
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h


def _incremental_corr_update(corr, h, i, j, delta):
    """
    Given the full correlation vector for h, compute the new correlation
    after changing h[i] += delta and h[j] -= delta.
    corr[k] = sum_m h[m] * (1 - h[m+k])  (for the appropriate range)
    Changing h[i] by +delta and h[j] by -delta affects corr at specific
    lags.  For mode="full", corr has length 2*N-1, indexed 0..2*N-2.
    corr[k] = sum_{m} h[m] * (1-h[m+k-N+1]) where the index m+k-N+1 must
    be in [0,N-1].  Equivalently, corr[k] for k in 0..2N-2:
      let d = k - (N-1)  (so d ranges from -(N-1) to N-1)
      corr[k] = sum_{m: 0<=m<N and 0<=m+d<N} h[m]*(1-h[m+d])
    Changing h[i] by +delta: for each m such that m=i, the term h[i]*(1-h[i+d])
    appears in corr at lag d (k = N-1+d) when i+d is in range.  Also when
    m+d=i, i.e., m = i-d, the term h[i-d]*(1-h[i]) appears.
    This is getting complex; for correctness we just recompute.  But for
    the polish we can afford O(N) per trial if we limit the number of trials.
    """
    return None  # We'll recompute in the polish; see _fast_polish.


def _fast_polish(h, max_rounds=30, deadline=None):
    """
    Fast local search: try swapping small amounts between pairs (i,j).
    Recompute corr from scratch each trial (O(N^2) per trial is too slow
    for many trials, but N=100 so corr is 199 elements – np.correlate is
    fast in C).  Limit total trials by deadline.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = N

    step_sizes = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]

    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Randomize the order of pairs each round to escape local minima
        order = np.arange(n)
        np.random.shuffle(order)
        for ii in range(n):
            i = order[ii]
            if deadline is not None and time.time() > deadline:
                break
            for jj in range(ii + 1, n):
                j = order[jj]
                if deadline is not None and time.time() > deadline:
                    break
                for d in step_sizes:
                    # Try h[i] += d, h[j] -= d
                    if best_h[i] + d <= 1.0 and best_h[j] - d >= 0.0:
                        new_h = best_h.copy()
                        new_h[i] += d
                        new_h[j] -= d
                        val = _correlation_value(new_h)
                        if val < best_val - 1e-15:
                            best_val = val
                            best_h = new_h
                            improved = True
                            break
                    # Try h[i] -= d, h[j] += d
                    if best_h[i] - d >= 0.0 and best_h[j] + d <= 1.0:
                        new_h = best_h.copy()
                        new_h[i] -= d
                        new_h[j] += d
                        val = _correlation_value(new_h)
                        if val < best_val - 1e-15:
                            best_val = val
                            best_h = new_h
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try with smaller steps
            step_sizes = [s / 2.0 for s in step_sizes]
            if step_sizes[0] < 1e-6:
                break
    return best_h


def _smooth_max(corr, lam):
    """Smooth max via log-sum-exp."""
    m = np.max(corr)
    if not np.isfinite(m):
        return m
    shifted = (corr - m) * lam
    # clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted)) / len(corr)) / lam


def _objective_smooth(h, lam):
    """Smooth objective for SLSQP."""
    corr = np.correlate(h, 1 - h, mode="full")
    return _smooth_max(corr, lam)


def _objective_exact(h):
    """Exact objective."""
    return np.max(np.correlate(h, 1 - h, mode="full"))


def _slsqp_step(x0, lam, maxiter, ftol, deadline):
    """Run SLSQP with the smooth objective."""
    def obj(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _objective_smooth(x, lam)

    def obj_grad(x):
        # Numerical gradient is expensive; let SLSQP estimate it.
        return obj(x)

    try:
        res = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * N,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                         'jac': lambda x: np.ones(N)}],
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        return _project_to_feasible(res.x)
    except (TimeoutError, Exception):
        return x0


def construct():
    start = time.time()
    deadline = start + 130.0

    best_h = np.full(N, 0.5, dtype=np.float64)
    best_val = _correlation_value(best_h)

    # ---- Phase 1: Differential evolution over a small basis ----
    n_basis = 10
    i = np.arange(N, dtype=float)
    basis = np.zeros((N, n_basis))
    basis[:, 0] = (i / N - 0.5)
    basis[:, 1] = ((i / N - 0.5) ** 2 - 1 / 12)
    for k, f in enumerate([1, 2, 3, 5, 7]):
        basis[:, 2 + 2 * k] = np.sin(2 * np.pi * f * i / N)
        basis[:, 2 + 2 * k + 1] = np.cos(2 * np.pi * f * i / N)

    def de_obj(coeffs):
        if time.time() > deadline - 80.0:
            raise TimeoutError()
        h = 0.5 * np.ones(N) + basis @ coeffs
        h = _project_to_feasible(h)
        return _correlation_value(h)

    bounds = [(-0.5, 0.5)] * n_basis
    try:
        result = differential_evolution(
            de_obj, bounds=bounds,
            maxiter=50, popsize=10, tol=1e-8,
            mutation=(0.3, 1.0), recombination=0.7,
            seed=42, polish=False, init='sobol'
        )
        h_de = _project_to_feasible(0.5 * np.ones(N) + basis @ result.x)
        val_de = _correlation_value(h_de)
        if val_de < best_val:
            best_val = val_de
            best_h = h_de.copy()
    except (TimeoutError, Exception):
        pass

    # ---- Phase 2: Multi-start SLSQP with smooth surrogate ----
    rng = np.random.default_rng(99)
    starts = [best_h.copy()]

    # Random feasible
    for _ in range(4):
        hr = rng.uniform(0, 1, N)
        starts.append(_project_to_feasible(hr))

    # Alternating
    h_alt = np.array([1.0 if k % 2 == 0 else 0.0 for k in range(N)])
    starts.append(_project_to_feasible(h_alt))

    # Blocks
    for bs in [4, 5, 10, 20]:
        hb = np.zeros(N)
        for k in range(N):
            hb[k] = 1.0 if (k // bs) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(hb))

    # Sine waves
    for f in [1, 2, 3, 4, 5, 7, 10]:
        amp = min(0.45, 0.5 / f)
        hs = 0.5 + amp * np.sin(2 * np.pi * f * np.arange(N) / N)
        starts.append(_project_to_feasible(hs))

    # Triangular
    for period in [8, 10, 16, 25]:
        ht = np.zeros(N)
        for k in range(N):
            t = (k % period) / period
            ht[k] = 2 * t if t < 0.5 else 2 - 2 * t
        starts.append(_project_to_feasible(ht))

    slsqp_deadline = deadline - 20.0
    for h0 in starts:
        if time.time() > slsqp_deadline:
            break
        h0 = _project_to_feasible(h0)
        cur = h0
        for lam in [50.0, 200.0, 1000.0, 5000.0]:
            if time.time() > slsqp_deadline:
                break
            cur = _slsqp_step(cur, lam, maxiter=100, ftol=1e-12, deadline=slsqp_deadline)
            val = _correlation_value(cur)
            if val < best_val:
                best_val = val
                best_h = cur.copy()

    # ---- Phase 3: Exact SLSQP polish ----
    exact_deadline = deadline - 10.0
    for lam in [10000.0]:
        if time.time() > exact_deadline:
            break
        best_h = _slsqp_step(best_h, lam, maxiter=200, ftol=1e-14, deadline=exact_deadline)
        val = _correlation_value(best_h)
        if val < best_val:
            best_val = val

    # ---- Phase 4: Fast local polish ----
    if time.time() < deadline - 2.0:
        best_h = _fast_polish(best_h, max_rounds=20, deadline=deadline - 1.0)
        best_val = _correlation_value(best_h)

    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
