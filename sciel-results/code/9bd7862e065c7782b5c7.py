"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _enforce_constraints(h, target_sum=None):
    """Enforce sum constraint and bounds via iterative projection (more robust)."""
    if target_sum is None:
        target_sum = N / 2.0
    h = h.copy()
    for _ in range(50):
        # Project onto sum constraint
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) < 1e-15:
            # Check bounds
            if np.all(h >= -1e-15) and np.all(h <= 1.0 + 1e-15):
                break
        h = h + diff / N
        # Project onto bounds
        h = np.clip(h, 0.0, 1.0)
    # Final exact fix
    s = np.sum(h)
    h = h + (target_sum - s) / N
    h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Start with 2-block solution as initial guess
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()

    def _compute_overlap(h):
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        return h * (1.0 - shifted)

    # Phase 1: SLSQP on raw max-overlap from the 2-block init
    try:
        def objective_raw(h):
            overlap = _compute_overlap(h)
            return float(np.max(overlap))

        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
        bounds = [(0.0, 1.0)] * N

        result = minimize(
            objective_raw,
            h0.copy(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
        )

        if result.fun < current_score:
            h_opt = _enforce_constraints(result.x.copy())
            new_score = _score(h_opt)
            if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                h_final = h_opt.copy()
                current_score = new_score

    except Exception:
        pass

    # Phase 2: Smooth surrogate with epigraph, moderate alpha for convergence
    try:
        if time.time() < deadline - 20.0:
            alpha = 300.0
            penalty = 5000.0

            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                smax = _smooth_max(overlap, alpha=alpha)
                return t + penalty * np.maximum(0.0, smax - t)**2

            x0 = np.concatenate([h_final, [current_score + 0.05]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum}
            ]

            result2 = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
            )

            if result2.fun < 10.0:
                h_opt2 = _enforce_constraints(result2.x[:N].copy())
                new_score2 = _score(h_opt2)
                if new_score2 < current_score and np.all(h_opt2 >= 0.0) and np.all(h_opt2 <= 1.0):
                    h_final = h_opt2.copy()
                    current_score = new_score2

    except Exception:
        pass

    # Phase 3: Tighter smooth polish with higher alpha
    try:
        if time.time() < deadline - 10.0:
            alpha = 800.0
            penalty = 20000.0

            def objective_polish(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                smax = _smooth_max(overlap, alpha=alpha)
                return t + penalty * np.maximum(0.0, smax - t)**2

            x0_polish = np.concatenate([h_final, [current_score + 0.005]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum}
            ]

            result3 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False}
            )

            if result3.fun < 10.0:
                h_opt3 = _enforce_constraints(result3.x[:N].copy())
                new_score3 = _score(h_opt3)
                if new_score3 < current_score and np.all(h_opt3 >= 0.0) and np.all(h_opt3 <= 1.0):
                    h_final = h_opt3.copy()
                    current_score = new_score3

    except Exception:
        pass

    # Phase 4: Limited multi-start with diverse inits (only if time permits)
    try:
        if time.time() < deadline - 25.0:
            inits = []

            # Alternating
            h_alt = np.zeros(N)
            h_alt[::2] = 0.7
            h_alt[1::2] = 0.3
            inits.append(h_alt)

            # Sine
            h_sin = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)
            inits.append(h_sin)

            # Cosine
            h_cos = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)
            inits.append(h_cos)

            # Random
            np.random.seed(42)
            h_rand = 0.5 + 0.3 * (np.random.rand(N) - 0.5)
            inits.append(h_rand)

            for h_init in inits:
                if time.time() > deadline - 12.0:
                    break

                h_init = _enforce_constraints(h_init.copy())
                init_score = _score(h_init)
                if init_score >= current_score:
                    continue

                def objective_raw(h):
                    overlap = _compute_overlap(h)
                    return float(np.max(overlap))

                constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
                bounds = [(0.0, 1.0)] * N

                result_r = minimize(
                    objective_raw,
                    h_init.copy(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 800, 'ftol': 1e-14, 'disp': False}
                )

                if result_r.fun < current_score:
                    h_opt_r = _enforce_constraints(result_r.x.copy())
                    new_score_r = _score(h_opt_r)
                    if new_score_r < current_score and np.all(h_opt_r >= 0.0) and np.all(h_opt_r <= 1.0):
                        h_final = h_opt_r.copy()
                        current_score = new_score_r

    except Exception:
        pass

    # Phase 5: Final ultra-tight polish
    try:
        if time.time() < deadline - 5.0:
            alpha = 1500.0
            penalty = 100000.0

            def objective_final(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                smax = _smooth_max(overlap, alpha=alpha)
                return t + penalty * np.maximum(0.0, smax - t)**2

            x0_final = np.concatenate([h_final, [current_score + 0.001]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum}
            ]

            result_final = minimize(
                objective_final,
                x0_final,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 5000, 'ftol': 1e-17, 'disp': False}
            )

            if result_final.fun < 10.0:
                h_opt_final = _enforce_constraints(result_final.x[:N].copy())
                new_score_final = _score(h_opt_final)
                if new_score_final < current_score and np.all(h_opt_final >= 0.0) and np.all(h_opt_final <= 1.0):
                    h_final = h_opt_final.copy()
                    current_score = new_score_final

    except Exception:
        pass

    return h_final
