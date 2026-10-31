"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

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
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) < 1e-15:
            if np.all(h >= -1e-15) and np.all(h <= 1.0 + 1e-15):
                break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    h = h + (target_sum - s) / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _compute_overlap(h):
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    return h * (1.0 - shifted)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Start with 2-block solution as initial guess
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()

    def _try_update(h_candidate):
        """Try to update best solution if candidate is better and valid."""
        nonlocal h_final, current_score
        h_opt = _enforce_constraints(h_candidate.copy())
        if not (np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0)):
            return False
        new_score = _score(h_opt)
        if new_score < current_score:
            h_final = h_opt.copy()
            current_score = new_score
            return True
        return False

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
            _try_update(result.x)

    except Exception:
        pass

    # Phase 2: Smooth epigraph formulation - minimize t s.t. t >= overlap_i
    # This is the key structural improvement: explicit epigraph variable
    try:
        if time.time() < deadline - 25.0:
            alpha = 200.0
            penalty = 5000.0

            def objective_epigraph(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                smax = _smooth_max(overlap, alpha=alpha)
                # Penalize t < smax
                return t + penalty * np.maximum(0.0, smax - t)**2

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                return t - np.max(overlap)

            x0 = np.concatenate([h_final, [current_score + 0.05]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result2 = minimize(
                objective_epigraph,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
            )

            if result2.fun < 100.0:
                _try_update(result2.x[:N])

    except Exception:
        pass

    # Phase 3: Differential evolution for global exploration
    # This is the structural change: use a global optimizer to escape local minima
    try:
        if time.time() < deadline - 30.0:
            def de_objective(h):
                # Penalty for constraint violation
                s = np.sum(h)
                penalty = 1000.0 * abs(s - N/2.0)
                overlap = _compute_overlap(h)
                return float(np.max(overlap)) + penalty

            bounds_de = [(0.0, 1.0)] * N

            # Run differential evolution with limited population
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=50,
                popsize=15,
                tol=1e-10,
                seed=42,
                polish=False,  # We'll polish separately
                update_interval=5,
                callback=None
            )

            if result_de.fun < current_score + 0.01:
                _try_update(result_de.x)

    except Exception:
        pass

    # Phase 4: High-precision smooth polish with epigraph
    try:
        if time.time() < deadline - 15.0:
            alpha = 500.0
            penalty = 20000.0

            def objective_polish(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                smax = _smooth_max(overlap, alpha=alpha)
                return t + penalty * np.maximum(0.0, smax - t)**2

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                return t - np.max(overlap)

            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result3 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False}
            )

            if result3.fun < 100.0:
                _try_update(result3.x[:N])

    except Exception:
        pass

    # Phase 5: Multi-start with structured inits + SLSQP polish
    try:
        if time.time() < deadline - 20.0:
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
                if time.time() > deadline - 10.0:
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
                    _try_update(result_r.x)

    except Exception:
        pass

    # Phase 6: Final ultra-tight polish
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

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                return t - np.max(overlap)

            x0_final = np.concatenate([h_final, [current_score + 0.001]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result_final = minimize(
                objective_final,
                x0_final,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 5000, 'ftol': 1e-17, 'disp': False}
            )

            if result_final.fun < 100.0:
                _try_update(result_final.x[:N])

    except Exception:
        pass

    return h_final
