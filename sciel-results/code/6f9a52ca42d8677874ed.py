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

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(30):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # The constant solution h[i] = 0.5 gives overlap = 0.5 * 0.5 = 0.25 everywhere
    # This is likely near-optimal or optimal for this problem
    h_const = np.full(N, 0.5)
    current_score = _score(h_const)  # should be 0.25
    h_final = h_const.copy()

    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Try to improve from constant solution using SLSQP with smooth max
    try:
        if time.time() < deadline - 30.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 10000.0 * np.maximum(0.0, smax - t)**2

            x0 = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False}
            )

            if result.fun < 100.0:
                _try_update(result.x[:N])

    except Exception:
        pass

    # Phase 2: Try L-BFGS-B with penalty for sum constraint from constant init
    try:
        if time.time() < deadline - 20.0:
            def objective_lbfgs(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=300.0)
                penalty = 1000.0 * (np.sum(h) - N/2.0)**2
                return smax + penalty

            bounds = [(0.0, 1.0)] * N
            result2 = minimize(
                objective_lbfgs,
                h_final.copy(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-16, 'disp': False}
            )
            _try_update(result2.x)

    except Exception:
        pass

    # Phase 3: Multi-start with diverse initializations
    try:
        if time.time() < deadline - 15.0:
            np.random.seed(123)
            n_starts = 10
            for trial in range(n_starts):
                if time.time() > deadline - 10.0:
                    break
                
                if trial == 0:
                    # Sine wave with various frequencies
                    freq = 1.0
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 1:
                    freq = 2.0
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 2:
                    freq = 3.0
                    h_rand = 0.5 + 0.2 * np.sin(2 * np.pi * freq * np.arange(N) / N)
                elif trial == 3:
                    # Two-frequency combination
                    h_rand = 0.5 + 0.2 * np.sin(2 * np.pi * np.arange(N) / N) + 0.1 * np.sin(4 * np.pi * np.arange(N) / N)
                elif trial == 4:
                    # Cosine
                    h_rand = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)
                elif trial == 5:
                    # Random smooth
                    h_rand = np.random.rand(N)
                    kernel = np.ones(7) / 7
                    h_rand = np.convolve(h_rand, kernel, mode='same')
                    h_rand = 0.2 + 0.6 * (h_rand - h_rand.min()) / (h_rand.max() - h_rand.min() + 1e-10)
                elif trial == 6:
                    # Block pattern
                    h_rand = np.tile([0.4, 0.6, 0.5], N // 3 + 1)[:N]
                elif trial == 7:
                    # Ramp
                    h_rand = 0.5 + 0.2 * (np.arange(N) / N - 0.5) * 2
                elif trial == 8:
                    # Alternating
                    h_rand = 0.5 + 0.2 * np.sin(np.pi * np.arange(N))
                else:
                    # Pure random
                    h_rand = np.random.rand(N)
                
                h_rand = np.clip(h_rand, 0.0, 1.0)
                h_rand = _enforce_sum(h_rand)
                
                # Quick SLSQP from this start
                def obj_ms(params):
                    h = params[:N]
                    t = params[N]
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    smax = _smooth_max(overlap, alpha=200.0)
                    return t + 5000.0 * np.maximum(0.0, smax - t)**2

                x0_ms = np.concatenate([h_rand, [_score(h_rand) + 0.05]])
                bounds_ms = [(0.0, 1.0)] * N + [(0.0, 1.0)]
                
                def cons_sum(params):
                    return np.sum(params[:N]) - N/2.0

                def cons_epi(params):
                    h = params[:N]
                    t = params[N]
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    return t - np.max(overlap)

                cons_ms = [
                    {'type': 'eq', 'fun': cons_sum},
                    {'type': 'ineq', 'fun': cons_epi}
                ]

                res_ms = minimize(
                    obj_ms,
                    x0_ms,
                    method='SLSQP',
                    bounds=bounds_ms,
                    constraints=cons_ms,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
                )

                if res_ms.fun < 100.0:
                    _try_update(res_ms.x[:N])

    except Exception:
        pass

    # Phase 4: Final polish with very tight tolerance
    try:
        if time.time() < deadline - 3.0:
            def obj_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=1000.0)
                return t + 50000.0 * np.maximum(0.0, smax - t)**2

            x0_p = np.concatenate([h_final, [current_score + 0.001]])
            bounds_p = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def cons_sum_p(params):
                return np.sum(params[:N]) - N/2.0

            def cons_epi_p(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)

            cons_p = [
                {'type': 'eq', 'fun': cons_sum_p},
                {'type': 'ineq', 'fun': cons_epi_p}
            ]

            res_p = minimize(
                obj_polish,
                x0_p,
                method='SLSQP',
                bounds=bounds_p,
                constraints=cons_p,
                options={'maxiter': 5000, 'ftol': 1e-18, 'disp': False}
            )

            if res_p.fun < 100.0:
                _try_update(res_p.x[:N])

    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
