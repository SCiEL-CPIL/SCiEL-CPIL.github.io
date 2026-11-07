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
K = 12  # number of control points
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
    cx = (np.arange(K) + 0.5) * (N / K)
    x = np.arange(N, dtype=float)
    return np.interp(x, cx, c)


def _enforce_sum(h, target=None):
    """Add a constant offset to make sum(h) == target, then clip and re-correct.
    Returns a feasible h (0<=h<=1, sum==target within tolerance).
    """
    if target is None:
        target = N / 2.0
    h = h.copy()
    diff = target - np.sum(h)
    h = h + diff / N
    for _ in range(50):
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
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
    shifted = corr - m
    return m + np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam


def _epigraph_objective(c, t, lam):
    """Epigraph formulation: minimize t + penalty for corr > t.
    This provides a smoother landscape than pure soft-max.
    """
    h = _feasible_from_control(c)
    corr = _correlation_array(h)
    # Penalty for exceeding t
    excess = np.maximum(0, corr - t)
    penalty = np.sum(np.exp(lam * excess)) / lam
    return t + penalty


def _objective_soft(c, lam):
    """Soft-max surrogate objective for control points c."""
    h = _feasible_from_control(c)
    corr = _correlation_array(h)
    return _soft_max(corr, lam)


def _objective_exact(c):
    h = _feasible_from_control(c)
    return _correlation_value(h)


def _local_polish_control(c, max_iters=60, step_sizes=None):
    """Local search on control points with sum re-enforcement."""
    best_c = np.array(c, dtype=float)
    best_val = _objective_exact(best_c)
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
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


def _local_polish_h(h, max_iters=50, step_sizes=None):
    """Local search directly on h (100-dim), maintaining sum via a partner swap.
    Focuses on positions near the max-correlation lag for efficiency."""
    best_h = np.array(h, dtype=float)
    best_val = _correlation_value(best_h)
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    n = len(best_h)
    for it in range(max_iters):
        # Identify the lag with max correlation
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_lag = np.argmax(corr)
        
        # Determine which h indices are involved in the max correlation
        # For lag k in "full" mode: corr[k] = sum(h[i] * (1-h[i+k-99])) for valid i
        # When k >= 99: corr[k] = sum(h[i] * (1-h[i-(k-99)])) for i from k-99 to 99
        # When k < 99: corr[k] = sum(h[i] * (1-h[i+(99-k)])) for i from 0 to 99-(99-k)
        k = max_lag - (n - 1)  # actual lag offset, can be negative
        if k >= 0:
            # corr[k] involves h[i] and h[i-k] for i in k..n-1
            involved = list(range(k, n))
        else:
            # corr[k] involves h[i] and h[i-k] for i in 0..n-1+k
            involved = list(range(0, n + k))
        
        # Try perturbing positions in the involved set first, then others
        priority_indices = sorted(set(involved) | set(range(n)))
        
        improved = False
        for i in priority_indices:
            if improved:
                break
            for d in step_sizes:
                if improved:
                    break
                for sign in (+1, -1):
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0.0 or new_h[i] > 1.0:
                        continue
                    needed = -sign * d
                    # Find a partner j to absorb the change
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
    de_deadline = start_time + 40.0
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
            maxiter=50,
            popsize=14,
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

    starts_c = []

    # From current best: bin averages
    bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                        for i in range(K)])
    starts_c.append(bin_avg.copy())

    # Random feasible control points
    for _ in range(12):
        starts_c.append(rng.uniform(0.0, 1.0, K))

    # Structured: alternating high/low
    for phase in (0, 1):
        c_alt = np.array([0.8 if (i + phase) % 2 == 0 else 0.2 for i in range(K)])
        starts_c.append(c_alt)

    # Structured: bimodal
    c_bi = np.linspace(0.8, 0.2, K)
    starts_c.append(c_bi.copy())
    starts_c.append(c_bi[::-1].copy())

    # Structured: oscillatory at several frequencies
    for freq in (1, 2, 3, 4, 5):
        t = np.arange(K)
        c_osc = 0.5 + 0.35 * np.sin(2 * np.pi * freq * t / K)
        starts_c.append(np.clip(c_osc, 0.0, 1.0))

    # Structured: triangular wave
    for phase in (0, 0.5):
        t = np.arange(K)
        c_tri = 0.5 + 0.35 * np.abs(2 * ((t + phase * K) / K) % 2 - 1) - 0.35
        starts_c.append(np.clip(c_tri, 0.0, 1.0))

    # Structured: sawtooth
    for phase in (0, 0.5):
        t = np.arange(K)
        c_saw = 0.5 + 0.35 * (2 * ((t + phase * K) / K) % 2 - 1)
        starts_c.append(np.clip(c_saw, 0.0, 1.0))

    # Structured: quadratic (bowl shape)
    t = np.arange(K)
    c_quad = 0.5 + 0.35 * ((2 * t / (K - 1) - 1) ** 2 - 1)
    starts_c.append(np.clip(c_quad, 0.0, 1.0))

    # Structured: plateau (high in middle, low at edges)
    c_plateau = np.ones(K) * 0.2
    c_plateau[K // 4:3 * K // 4] = 0.8
    starts_c.append(c_plateau.copy())
    c_plateau2 = np.ones(K) * 0.8
    c_plateau2[K // 4:3 * K // 4] = 0.2
    starts_c.append(c_plateau2.copy())

    slsqp_deadline = start_time + 135.0

    for idx, c_start in enumerate(starts_c):
        if time.time() > slsqp_deadline - 12.0:
            break
        _check()

        current_c = np.clip(np.asarray(c_start, dtype=float), 0.0, 1.0)

        # Ramping lambda soft-max optimization
        for lam in (10.0, 50.0, 200.0, 800.0, 2000.0, 5000.0, 10000.0):
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
                    options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14}
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

        # Final exact polish on control points
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
                options={'maxiter': 1500, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            c_e = np.clip(res_e.x, 0.0, 1.0)
            h_e = _feasible_from_control(c_e)
            v_e = _correlation_value(h_e)
            if v_e < best_val:
                best_val = v_e
                best_h = h_e.copy()
        except Exception:
            pass

        # Epigraph refinement: introduce t and minimize t + penalty
        if time.time() > slsqp_deadline - 2.0:
            break
        try:
            def epigraph_obj(x):
                c_x = x[:K]
                t = x[K]
                h = _feasible_from_control(c_x)
                corr = _correlation_array(h)
                excess = np.maximum(0, corr - t)
                lam_ep = 500.0
                penalty = np.sum(np.exp(lam_ep * excess)) / lam_ep
                return t + penalty

            # Initialize t with current best value
            t_init = _correlation_value(_feasible_from_control(current_c))
            x_init = np.concatenate([current_c, [t_init]])
            bounds_ep = [(0.0, 1.0)] * K + [(0.0, 2.0)]

            res_ep = minimize(
                epigraph_obj,
                x_init,
                method='L-BFGS-B',
                bounds=bounds_ep,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            c_ep = np.clip(res_ep.x[:K], 0.0, 1.0)
            h_ep = _feasible_from_control(c_ep)
            v_ep = _correlation_value(h_ep)
            if v_ep < best_val:
                best_val = v_ep
                best_h = h_ep.copy()
                current_c = c_ep
        except Exception:
            pass

    _check()

    # ---- Phase 3: Local polish on control points, then on h ----
    if time.time() < start_time + DEADLINE - 8.0:
        try:
            bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                                for i in range(K)])
            c_pol = _local_polish_control(bin_avg, max_iters=80)
            h_pol = _feasible_from_control(c_pol)
            v_pol = _correlation_value(h_pol)
            if v_pol < best_val:
                best_val = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if time.time() < start_time + DEADLINE - 3.0:
        try:
            best_h = _local_polish_h(best_h, max_iters=60)
            v = _correlation_value(best_h)
            if v < best_val:
                best_val = v
        except Exception:
            pass

    # ---- Final feasibility guarantee ----
    best_h = _enforce_sum(best_h)
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
