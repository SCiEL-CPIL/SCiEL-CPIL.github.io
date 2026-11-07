"""Piecewise-linear control point parameterization for Erdos minimum overlap.

The 100 step heights are represented as a linear interpolation of K control
points. This low-dimensional smooth parameterization lets the optimizer
discover global shapes (bimodal, oscillatory) that are hard to reach when
optimizing 100 independent variables. A smooth offset enforces the sum
constraint. A log-sum-exp soft-max surrogate with ramping lambda sharpens
the nonsmooth max-correlation objective for L-BFGS-B.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
K = 10  # number of control points
DEADLINE = 145.0  # safety margin before 150s kill


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _correlation_array(h):
    return np.correlate(h, 1 - h, mode="full")


def _interp_from_control(c):
    """Linearly interpolate K control points to N samples.
    Control points are placed at positions (i + 0.5) * N / K for i in 0..K-1,
    i.e., at the centers of K equal bins. We interpolate at the N sample
    positions.
    """
    c = np.asarray(c, dtype=float)
    # Control point x-positions: centers of K bins
    cx = (np.arange(K) + 0.5) * (N / K)
    # Sample x-positions: 0, 1, ..., N-1
    x = np.arange(N, dtype=float)
    # np.interp clamps to endpoints for out-of-range
    return np.interp(x, cx, c)


def _enforce_sum(h, target=None):
    """Add a constant offset to make sum(h) == target, then clip and re-correct.
    Returns a feasible h (0<=h<=1, sum==target within tolerance).
    """
    if target is None:
        target = N / 2.0
    h = h.copy()
    # First pass: offset
    diff = target - np.sum(h)
    h = h + diff / N
    # Clip and iteratively re-correct
    for _ in range(50):
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        # Adjust only elements that can absorb
        if diff > 0:
            mask = h < 1.0 - 1e-12
        else:
            mask = h > 1e-12
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
    return h


def _feasible_from_control(c, target=None):
    """Interpolate control points, then enforce sum and bounds."""
    h = _interp_from_control(c)
    return _enforce_sum(h, target)


def _soft_max(corr, lam):
    """Soft max via log-sum-exp, numerically stable."""
    m = np.max(corr)
    if lam <= 0:
        return m
    # log-sum-exp
    shifted = corr - m
    return m + np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam


def _objective_soft(c, lam):
    """Soft-max surrogate objective for control points c."""
    h = _feasible_from_control(c)
    corr = _correlation_array(h)
    return _soft_max(corr, lam)


def _objective_exact(c):
    h = _feasible_from_control(c)
    return _correlation_value(h)


def _local_polish_control(c, max_iters=40, step_sizes=None):
    """Local search on control points with sum re-enforcement."""
    best_c = np.array(c, dtype=float)
    best_val = _objective_exact(best_c)
    if step_sizes is None:
        step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    n = len(best_c)
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for d in step_sizes:
                for sign in (+1, -1):
                    new_c = best_c.copy()
                    new_c[i] += sign * d
                    val = _objective_exact(new_c)
                    if val < best_val - 1e-13:
                        best_val = val
                        best_c = new_c
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_c


def _local_polish_h(h, max_iters=30, step_sizes=None):
    """Local search directly on h (100-dim), maintaining sum via a partner swap."""
    best_h = np.array(h, dtype=float)
    best_val = _correlation_value(best_h)
    if step_sizes is None:
        step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1]
    n = len(best_h)
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for d in step_sizes:
                for sign in (+1, -1):
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0.0 or new_h[i] > 1.0:
                        continue
                    # Find partner j to absorb the change
                    needed = -sign * d
                    # Try a few candidate partners spread out
                    cands = [j for j in range(n) if j != i]
                    found = False
                    for j in cands:
                        val_j = best_h[j] + needed
                        if 0.0 <= val_j <= 1.0:
                            new_h[j] = val_j
                            val = _correlation_value(new_h)
                            if val < best_val - 1e-13:
                                best_val = val
                                best_h = new_h
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_h


def construct():
    start_time = time.time()

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    def _elapsed():
        return time.time() - start_time

    def _check():
        if _elapsed() > DEADLINE:
            raise TimeoutError()

    # ---- Phase 1: Differential evolution over K control points ----
    de_deadline = start_time + 45.0
    try:
        c_lo = np.full(K, 0.0)
        c_hi = np.full(K, 1.0)

        def de_obj(c):
            if time.time() > de_deadline:
                raise TimeoutError()
            return _objective_soft(c, lam=20.0)

        res_de = differential_evolution(
            de_obj,
            bounds=list(zip(c_lo, c_hi)),
            maxiter=60,
            popsize=12,
            tol=1e-7,
            mutation=(0.4, 1.0),
            recombination=0.7,
            seed=7,
            polish=False,
            init='sobol'
        )
        if res_de.x is not None:
            h_de = _feasible_from_control(res_de.x)
            v_de = _correlation_value(h_de)
            if v_de < best_val:
                best_val = v_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    _check()

    # ---- Phase 2: Multi-start L-BFGS-B with ramping lambda soft-max ----
    rng = np.random.default_rng(2024)

    # Build diverse starting control vectors
    starts_c = []

    # From current best: fit control points by least squares (smooth init)
    # Use a simple projection: bin averages of best_h
    bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                        for i in range(K)])
    starts_c.append(bin_avg.copy())

    # Random feasible control points
    for _ in range(10):
        starts_c.append(rng.uniform(0.0, 1.0, K))

    # Structured: alternating high/low
    for phase in (0, 1):
        c_alt = np.array([0.8 if (i + phase) % 2 == 0 else 0.2 for i in range(K)])
        starts_c.append(c_alt)

    # Structured: bimodal (high at ends, low in middle)
    c_bi = np.linspace(0.8, 0.2, K)
    starts_c.append(c_bi.copy())
    starts_c.append(c_bi[::-1].copy())

    # Structured: oscillatory at several frequencies
    for freq in (1, 2, 3, 4):
        t = np.arange(K)
        c_osc = 0.5 + 0.35 * np.sin(2 * np.pi * freq * t / K)
        starts_c.append(np.clip(c_osc, 0.0, 1.0))

    slsqp_deadline = start_time + 135.0

    for idx, c_start in enumerate(starts_c):
        if time.time() > slsqp_deadline - 12.0:
            break
        _check()

        current_c = np.clip(np.asarray(c_start, dtype=float), 0.0, 1.0)

        # Ramping lambda soft-max optimization
        for lam in (10.0, 50.0, 200.0, 800.0):
            if time.time() > slsqp_deadline - 5.0:
                break

            def obj(x, lam=lam):
                return _objective_soft(x, lam)

            try:
                res = minimize(
                    obj,
                    current_c,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * K,
                    options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                c_opt = np.clip(res.x, 0.0, 1.0)
                h_opt = _feasible_from_control(c_opt)
                v = _correlation_value(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                current_c = c_opt
            except Exception:
                pass

        # Final exact (non-smooth) polish on control points
        if time.time() > slsqp_deadline - 4.0:
            break
        try:
            def obj_exact(x):
                return _objective_exact(x)

            res_e = minimize(
                obj_exact,
                current_c,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * K,
                options={'maxiter': 600, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            c_e = np.clip(res_e.x, 0.0, 1.0)
            h_e = _feasible_from_control(c_e)
            v_e = _correlation_value(h_e)
            if v_e < best_val:
                best_val = v_e
                best_h = h_e.copy()
        except Exception:
            pass

    _check()

    # ---- Phase 3: Local polish on control points, then on h ----
    if time.time() < start_time + DEADLINE - 8.0:
        try:
            # Fit control points from best_h for a good local start
            bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                                for i in range(K)])
            c_pol = _local_polish_control(bin_avg, max_iters=50)
            h_pol = _feasible_from_control(c_pol)
            v_pol = _correlation_value(h_pol)
            if v_pol < best_val:
                best_val = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if time.time() < start_time + DEADLINE - 3.0:
        try:
            best_h = _local_polish_h(best_h, max_iters=25)
            v = _correlation_value(best_h)
            if v < best_val:
                best_val = v
        except Exception:
            pass

    # ---- Final feasibility guarantee ----
    best_h = _enforce_sum(best_h)
    # Ensure finite
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
