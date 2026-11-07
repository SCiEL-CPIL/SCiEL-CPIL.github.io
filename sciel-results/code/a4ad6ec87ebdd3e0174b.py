"""Erdos minimum overlap - piecewise-linear control-point parameterization
with soft-max surrogate and epigraph optimization.

Strategy:
- Represent h as a piecewise-linear function defined by K control points.
- Linearly interpolate to N=100 samples, then project to satisfy sum=N/2 and 0<=h<=1.
- Use a soft-max (log-sum-exp) surrogate for the max-correlation objective.
- Use an epigraph formulation with an auxiliary variable t.
- Multi-start optimization with differential evolution on control points.
- Final local polish in the full 100-dim space.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
K = 12  # number of control points


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _correlation_array(h):
    """Return the full correlation array."""
    return np.correlate(h, 1 - h, mode="full")


def _interp_from_control(cp):
    """
    Interpolate K control points to N samples.
    cp has K values, placed at evenly spaced positions, linearly interpolated to N points.
    """
    cp = np.asarray(cp, dtype=float)
    # Control points are at positions 0, N/(K-1), 2*N/(K-1), ..., N-1
    # But we want them to define a piecewise linear function over [0, N-1]
    # Place K control points at indices 0, N/(K-1), ..., N-1
    if K == 1:
        return np.full(N, cp[0])
    ctrl_idx = np.linspace(0, N - 1, K)
    h = np.interp(np.arange(N), ctrl_idx, cp)
    return h


def _enforce_sum(h, target_sum=None):
    """Adjust h to have the correct sum while keeping 0<=h<=1."""
    if target_sum is None:
        target_sum = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - target_sum
    for _ in range(50):
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
        diff = np.sum(h) - target_sum
    return h


def _feasible_from_control(cp):
    """Convert control points to a feasible h vector."""
    h = _interp_from_control(cp)
    h = _enforce_sum(h)
    return h


def _feasible_from_h(h):
    """Project h to feasible set."""
    return _enforce_sum(h)


def _soft_max(corr, lam):
    """Soft max via log-sum-exp."""
    m = np.max(corr)
    if m > 1e15:
        return m
    return m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam


def _objective_soft(cp, lam):
    """Soft-max objective from control points."""
    h = _feasible_from_control(cp)
    corr = _correlation_array(h)
    return _soft_max(corr, lam)


def _objective_exact(cp):
    """Exact max-correlation objective from control points."""
    h = _feasible_from_control(cp)
    corr = _correlation_array(h)
    return np.max(corr)


def _epigraph_solve(cp_init, deadline):
    """
    Epigraph formulation: minimize t subject to corr[k] <= t for all k,
    0 <= h <= 1, sum(h) = N/2.
    Variables: [h_0, ..., h_{N-1}, t]
    """
    x0 = np.concatenate([cp_init, [0.0]])  # h + t
    n_vars = N + 1

    def objective(x):
        t = x[-1]
        h = x[:N]
        # Penalty for constraint violation
        corr = _correlation_array(h)
        viol = np.max(corr) - t
        return t + 100.0 * max(0, viol) ** 2

    def objective_grad(x):
        h = x[:N]
        t = x[-1]
        corr = _correlation_array(h)
        viol = np.max(corr) - t
        grad_t = 1.0 - 200.0 * max(0, viol)
        grad_h = -200.0 * max(0, viol) * (corr - t) / np.max(corr) if np.max(corr) > 0 else 0
        return np.concatenate([grad_h, [grad_t]])

    bounds = [(0.0, 1.0)] * N + [(-1.0, 1.0)]
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        h_opt = _enforce_sum(res.x[:N])
        return h_opt
    except Exception:
        return cp_init.copy()


def _local_polish_h(h, max_iters=30, step_sizes=None):
    """Local search in full 100-dim space."""
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)

    if step_sizes is None:
        step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    for it in range(max_iters):
        improved = False
        for i in range(n):
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0 or new_h[i] > 1:
                        continue
                    needed = -sign * d
                    # Find best j to compensate
                    best_j = -1
                    best_val_j = best_val + 1e-10
                    for j in range(n):
                        if j == i:
                            continue
                        val_j = best_h[j] + needed
                        if 0 <= val_j <= 1:
                            new_h[j] = val_j
                            v = _correlation_value(new_h)
                            if v < best_val_j:
                                best_val_j = v
                                best_j = j
                    if best_j >= 0 and best_val_j < best_val - 1e-13:
                        best_val = best_val_j
                        best_h = new_h.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_h


def _local_polish_control(cp, max_iters=20, step_sizes=None):
    """Local search in control-point space."""
    best_cp = cp.copy()
    best_val = _objective_exact(best_cp)

    if step_sizes is None:
        step_sizes = [0.01, 0.02, 0.05, 0.1, 0.2]

    for it in range(max_iters):
        improved = False
        for k in range(len(best_cp)):
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_cp = best_cp.copy()
                    new_cp[k] += sign * d
                    new_cp = np.clip(new_cp, 0.0, 1.0)
                    v = _objective_exact(new_cp)
                    if v < best_val - 1e-13:
                        best_val = v
                        best_cp = new_cp.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_cp


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Start with uniform
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    best_cp = np.full(K, 0.5)

    # Phase 1: Differential evolution on control points with soft-max ramping
    n_cp = K
    bounds = [(0.0, 1.0)] * n_cp

    de_deadline = start_time + 60.0

    def de_objective(cp):
        if time.time() > de_deadline:
            raise TimeoutError()
        # Use a moderate lambda for global exploration
        return _objective_soft(cp, lam=50.0)

    try:
        result = differential_evolution(
            de_objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-7,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,
            init='sobol'
        )
        if result.x is not None:
            h_de = _feasible_from_control(result.x)
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
                best_cp = result.x.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    # Phase 2: Multi-start SLSQP with soft-max ramping in control-point space
    rng = np.random.default_rng(123)
    slsqp_deadline = start_time + 120.0

    # Generate diverse starting points in control-point space
    starts = [best_cp.copy()]

    # Random starts
    for _ in range(6):
        cp_rand = rng.uniform(0, 1, K)
        starts.append(cp_rand)

    # Structured: bimodal (high-low-high-low pattern)
    for pattern in [
        np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2][:K]),
        np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8][:K]),
        np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1][:K]),
        np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9][:K]),
    ]:
        starts.append(pattern.copy())

    # Sine wave patterns
    for freq in [1, 2, 3]:
        amp = 0.35
        t = np.linspace(0, 2 * np.pi * freq, K)
        cp_sin = 0.5 + amp * np.sin(t)
        starts.append(np.clip(cp_sin, 0, 1))

    # Triangular patterns
    for period in [4, 6, 8]:
        cp_tri = np.array([
            0.5 + 0.4 * (2 * abs((i % period) / period - 0.5) - 0.5)
            for i in range(K)
        ])
        starts.append(np.clip(cp_tri, 0, 1))

    # Optimize from each start
    for idx, cp_start in enumerate(starts):
        if time.time() > slsqp_deadline - 15.0:
            break

        cp_start = np.clip(cp_start, 0.0, 1.0)
        current_cp = cp_start.copy()

        # Ramp lambda from low to high
        for lam in [10.0, 50.0, 200.0, 1000.0]:
            if time.time() > slsqp_deadline - 5.0:
                break

            def soft_obj(cp, lam=lam):
                h = _feasible_from_control(cp)
                corr = _correlation_array(h)
                return _soft_max(corr, lam)

            try:
                res = minimize(
                    soft_obj,
                    current_cp,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * K,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                cp_opt = np.clip(res.x, 0.0, 1.0)
                h_opt = _feasible_from_control(cp_opt)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    best_cp = cp_opt.copy()
                current_cp = cp_opt
            except Exception:
                pass

        # Final exact optimization in control-point space
        if time.time() > slsqp_deadline - 3.0:
            break

        try:
            def exact_obj(cp):
                h = _feasible_from_control(cp)
                corr = _correlation_array(h)
                return np.max(corr)

            res_exact = minimize(
                exact_obj,
                current_cp,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * K,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            cp_exact = np.clip(res_exact.x, 0.0, 1.0)
            h_exact = _feasible_from_control(cp_exact)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
                best_cp = cp_exact.copy()
        except Exception:
            pass

    # Phase 3: Local polish in control-point space
    if time.time() < deadline - 10.0:
        try:
            best_cp = _local_polish_control(best_cp, max_iters=30)
            h_cp = _feasible_from_control(best_cp)
            val_cp = _correlation_value(h_cp)
            if val_cp < best_val:
                best_val = val_cp
                best_h = h_cp.copy()
        except Exception:
            pass

    # Phase 4: Local polish in full 100-dim space
    if time.time() < deadline - 5.0:
        try:
            best_h = _local_polish_h(best_h, max_iters=25)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # Final projection
    best_h = _enforce_sum(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
