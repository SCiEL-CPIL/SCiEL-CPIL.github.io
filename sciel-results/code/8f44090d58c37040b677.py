"""Model-and-optimize approach for Erdos minimum overlap.

Key insight: the max correlation value is driven by the peaks of the
correlation function. We can parameterize h as a smooth function of a
few basis coefficients, then directly optimize those parameters.

Strategy:
1. Differential evolution over basis coefficients to find a good global structure
2. Multi-start SLSQP with ramping soft-max surrogate for local refinement
3. Pairwise local search with better partner selection
4. High-precision polish with L-BFGS-B (faster) + SLSQP with epigraph
"""

N = 100
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

DEADLINE_OFFSET = 143.0  # Safety margin for 150s limit


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0).copy()
    diff = np.sum(h) - N / 2.0
    for _ in range(30):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _construct_from_basis(coeffs):
    """
    Build h from a basis expansion:
    h[i] = 0.5 + sum_k coeffs[k] * basis_k(i)
    """
    n = len(coeffs)
    i = np.arange(N, dtype=float)
    h = np.full(N, 0.5)

    if n > 0:
        h += coeffs[0] * (i / N - 0.5)
    if n > 1:
        h += coeffs[1] * ((i / N - 0.5) ** 2 - 1 / 12)
    if n > 2:
        h += coeffs[2] * np.sin(2 * np.pi * i / N)
    if n > 3:
        h += coeffs[3] * np.cos(2 * np.pi * i / N)
    if n > 4:
        h += coeffs[4] * np.sin(4 * np.pi * i / N)
    if n > 5:
        h += coeffs[5] * np.cos(4 * np.pi * i / N)
    if n > 6:
        h += coeffs[6] * np.sin(6 * np.pi * i / N)
    if n > 7:
        h += coeffs[7] * np.cos(6 * np.pi * i / N)
    if n > 8:
        h += coeffs[8] * np.sin(10 * np.pi * i / N)
    if n > 9:
        h += coeffs[9] * np.cos(10 * np.pi * i / N)
    if n > 10:
        h += coeffs[10] * np.sin(14 * np.pi * i / N)
    if n > 11:
        h += coeffs[11] * np.cos(14 * np.pi * i / N)

    return h


def _soft_max_obj(x, lam):
    """
    Smooth surrogate: soft-max of correlation values.
    Returns the soft-max value (not divided by N*2 since we minimize).
    """
    h = np.clip(x, 0.0, 1.0)
    corr = np.correlate(h, 1.0 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    shifted = corr - m
    return m + np.log(np.sum(np.exp(shifted)) / len(corr))


def _local_polish(h, max_iters=50, time_budget=20.0):
    """
    Local search: try pairwise swaps (increase one, decrease another)
    to maintain sum constraint. Better partner selection based on
    correlation impact.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    start_t = time.time()

    step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    for it in range(max_iters):
        if time.time() - start_t > time_budget:
            break
        improved = False
        # Randomize element order to avoid bias
        order = np.random.default_rng(it * 1000 + 7).permutation(n)
        for i in order:
            if improved:
                break
            for d in step_sizes:
                if improved:
                    break
                for sign in [+1, -1]:
                    if improved:
                        break
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < -1e-12 or new_h[i] > 1.0 + 1e-12:
                        continue
                    new_h[i] = np.clip(new_h[i], 0.0, 1.0)
                    # Need to adjust another element to keep sum
                    needed = -(new_h[i] - best_h[i])
                    # Try to find best partner: pick j that minimizes new value
                    # Heuristic: try a few candidates (nearest and farthest)
                    candidates = []
                    for j in range(n):
                        if j == i:
                            continue
                        val_j = best_h[j] + needed
                        if -1e-12 <= val_j <= 1.0 + 1e-12:
                            # Score: prefer larger distance (less local correlation impact)
                            dist = abs(i - j)
                            # Also prefer partners near 0.5 (more room)
                            room = min(val_j, 1.0 - val_j)
                            candidates.append((dist + 10 * room, j))
                    if not candidates:
                        continue
                    # Sort by preference, try top few
                    candidates.sort(reverse=True)
                    tried = 0
                    for _, j in candidates[:8]:
                        if tried >= 4:
                            break
                        tried += 1
                        new_h[j] = np.clip(best_h[j] + needed, 0.0, 1.0)
                        # Check sum
                        if abs(np.sum(new_h) - N / 2.0) > 1e-10:
                            continue
                        val = _correlation_value(new_h)
                        if val < best_val - 1e-13:
                            best_val = val
                            best_h = new_h
                            improved = True
                            break
        if not improved:
            break

    return best_h


def _epigraph_slsqp(h_start, time_budget=15.0):
    """
    Epigraph formulation: minimize t subject to
    corr_k <= t for all k, sum(h) = N/2, 0 <= h <= 1.
    This is a smooth optimization with 100+N variables and 199+N constraints.
    """
    start_t = time.time()

    def epigraph_obj(vars):
        h = vars[:N]
        t = vars[N]
        # Penalty for violations
        corr = np.correlate(h, 1.0 - h, mode="full")
        violations = np.maximum(corr - t, 0.0)
        return t + 1000.0 * np.sum(violations ** 2)

    def epigraph_grad(vars):
        h = vars[:N]
        t = vars[N]
        corr = np.correlate(h, 1.0 - h, mode="full")
        # Gradient of t w.r.t. t
        grad_t = 1.0
        # Gradient of t w.r.t. h: through correlation
        # d(corr_k)/dh_i is complex; use numerical
        eps = 1e-7
        grad_h = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            corr_plus = np.correlate(h_plus, 1.0 - h_plus, mode="full")
            viol_plus = np.maximum(corr_plus - t, 0.0)
            h_minus = h.copy()
            h_minus[i] -= eps
            corr_minus = np.correlate(h_minus, 1.0 - h_minus, mode="full")
            viol_minus = np.maximum(corr_minus - t, 0.0)
            grad_h[i] = (np.sum(viol_plus ** 2) - np.sum(viol_minus ** 2)) / (2 * eps) * 1000.0
        grad = np.zeros(N + 1)
        grad[:N] = grad_h
        grad[N] = grad_t
        return grad

    def sum_constraint(vars):
        return np.sum(vars[:N]) - N / 2.0

    def sum_jac(vars):
        j = np.zeros(N + 1)
        j[:N] = 1.0
        return j

    x0 = np.zeros(N + 1)
    x0[:N] = h_start
    x0[N] = _correlation_value(h_start) * N / 2  # initial t

    bounds = [(0.0, 1.0)] * N + [(-1.0, 5.0)]
    constraints = [{'type': 'eq', 'fun': sum_constraint, 'jac': sum_jac}]

    try:
        res = minimize(
            epigraph_obj,
            x0,
            jac=epigraph_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
        )
        h_opt = _project_to_feasible(res.x[:N])
        val = _correlation_value(h_opt)
        return h_opt, val
    except Exception:
        return h_start, _correlation_value(h_start)


def _l_bfgs_polish(h_start, time_budget=15.0):
    """
    L-BFGS-B polish with soft-max surrogate and ramping lambda.
    L-BFGS-B is faster than SLSQP for problems with simple bounds.
    """
    start_t = time.time()
    current_h = h_start.copy()
    best_h = h_start.copy()
    best_val = _correlation_value(best_h)

    for lam in [20.0, 50.0, 100.0, 200.0, 500.0]:
        if time.time() - start_t > time_budget - 2.0:
            break

        def obj(x, lam=lam):
            h = np.clip(x, 0.0, 1.0)
            corr = np.correlate(h, 1.0 - h, mode="full")
            m = np.max(corr)
            shifted = corr - m
            lse = m + np.log(np.sum(np.exp(shifted)) / len(corr))
            # Add a small penalty for sum violation (L-BFGS-B doesn't handle eq constraints well)
            sum_pen = 100.0 * (np.sum(h) - N / 2.0) ** 2
            return lse + sum_pen

        def obj_grad(x, lam=lam):
            h = np.clip(x, 0.0, 1.0)
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros(N)
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] = min(1.0, h[i] + eps)
                h_minus = h.copy()
                h_minus[i] = max(0.0, h[i] - eps)
                corr_p = np.correlate(h_plus, 1.0 - h_plus, mode="full")
                corr_m = np.correlate(h_minus, 1.0 - h_minus, mode="full")
                mp = np.max(corr_p)
                mm = np.max(corr_m)
                sp = mp + np.log(np.sum(np.exp(corr_p - mp)) / len(corr_p))
                sm = mm + np.log(np.sum(np.exp(corr_m - mm)) / len(corr_m))
                f_plus = sp + 100.0 * (np.sum(h_plus) - N / 2.0) ** 2
                f_minus = sm + 100.0 * (np.sum(h_minus) - N / 2.0) ** 2
                grad[i] = (f_plus - f_minus) / (h_plus[i] - h_minus[i] + 1e-18)
            return grad

        try:
            res = minimize(
                obj,
                current_h,
                jac=obj_grad,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
            current_h = h_opt
        except Exception:
            pass

    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    # Phase 1: Differential evolution over basis coefficients
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    de_deadline = start_time + 40.0

    def de_objective(coeffs):
        if time.time() > de_deadline:
            return 999.0
        h = _construct_from_basis(coeffs)
        h = _project_to_feasible(h)
        return _correlation_value(h)

    try:
        result = differential_evolution(
            de_objective,
            bounds=bounds,
            maxiter=60,
            popsize=10,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            init='sobol'
        )
        if result.x is not None and result.fun < best_val:
            h_de = _project_to_feasible(_construct_from_basis(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except Exception:
        pass

    # Phase 2: Multi-start L-BFGS-B + SLSQP with soft-max surrogate
    rng = np.random.default_rng(123)
    starts = [best_h.copy()]

    # Random feasible starts
    for _ in range(8):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)

    # Structured: alternating
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))

    # Blocks
    for block_size in [3, 4, 5, 7, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))

    # Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10, 13]:
        amp = min(0.45, 0.5 / max(1, freq * 0.3))
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))

    # Triangular
    for period in [8, 10, 12, 20, 25]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))

    slsqp_deadline = start_time + 110.0

    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 20.0:
            break

        h_start = _project_to_feasible(h_start)
        current_h = h_start

        # L-BFGS-B with ramping lambda (faster than SLSQP)
        for lam in [30.0, 100.0, 300.0]:
            if time.time() > slsqp_deadline - 15.0:
                break

            def smooth_obj(x, lam=lam):
                h = np.clip(x, 0.0, 1.0)
                corr = np.correlate(h, 1.0 - h, mode="full")
                m = np.max(corr)
                shifted = corr - m
                lse = m + np.log(np.sum(np.exp(shifted)) / len(corr))
                return lse

            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass

        # Final exact SLSQP
        if time.time() > slsqp_deadline - 10.0:
            break

        try:
            def exact_obj(x):
                h = np.clip(x, 0.0, 1.0)
                corr = np.correlate(h, 1.0 - h, mode="full")
                return np.max(corr)

            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 400, 'ftol': 1e-15}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass

    # Phase 3: Local search polish
    if time.time() < deadline - 35.0:
        try:
            best_h = _local_polish(best_h, max_iters=50, time_budget=20.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # Phase 4: L-BFGS-B high-precision polish
    remaining = deadline - time.time() - 5.0
    if remaining > 15.0:
        try:
            best_h, best_val = _l_bfgs_polish(best_h, time_budget=min(remaining * 0.5, 20.0))
        except Exception:
            pass

    # Phase 5: Epigraph SLSQP on the best result
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        try:
            h_epi, val_epi = _epigraph_slsqp(best_h, time_budget=min(remaining * 0.6, 15.0))
            if val_epi < best_val:
                best_val = val_epi
                best_h = h_epi.copy()
        except Exception:
            pass

    # Final: one more local polish
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        try:
            best_h = _local_polish(best_h, max_iters=20, time_budget=min(remaining, 10.0))
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
