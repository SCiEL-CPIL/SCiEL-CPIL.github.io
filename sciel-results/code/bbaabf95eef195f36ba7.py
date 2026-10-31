"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h over all shifts."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100

def _max_overlap_full(h):
    """Compute max overlap of h with shifted (1-h) over ALL shifts k=0..N-1."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        if k == 0:
            shifted = one_minus_h
        else:
            shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return float(max_ov)

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _all_overlaps(h):
    """Compute all N overlap values for smooth max."""
    one_minus_h = 1.0 - h
    overlaps = np.zeros(N)
    for k in range(N):
        if k == 0:
            shifted = one_minus_h
        else:
            shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    return overlaps

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-15:
        # Distribute residual to elements not at bounds
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Initial guess: 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    h0 = _enforce_sum(h0)
    current_score = _max_overlap_full(h0)
    h_final = h0.copy()

    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-9:
                sc = _max_overlap_full(h_c)
                if sc < current_score - 1e-14:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Differential Evolution for global exploration
    try:
        if time.time() < deadline - 60.0:
            def de_objective(h):
                h = _enforce_sum(h)
                overlaps = _all_overlaps(h)
                return _smooth_max(overlaps, alpha=50.0)

            bounds = [(0.0, 1.0)] * N

            # Use DE with moderate population for global search
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.2),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )

            if de_result.fun < current_score:
                _try_update(de_result.x)

    except Exception:
        pass

    # Phase 2: SLSQP polish on raw max from DE result
    try:
        if time.time() < deadline - 30.0:
            def objective_raw(h):
                overlaps = _all_overlaps(h)
                return float(np.max(overlaps))

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_raw,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
            )

            if result.fun < current_score:
                _try_update(result.x)

    except Exception:
        pass

    # Phase 3: Smooth epigraph SLSQP for further refinement
    try:
        if time.time() < deadline - 20.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                overlaps = _all_overlaps(h)
                smax = _smooth_max(overlaps, alpha=300.0)
                return t + 5000.0 * np.maximum(0.0, smax - t)**2

            x0 = np.concatenate([h_final, [current_score + 0.05]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                overlaps = _all_overlaps(h)
                return t - np.max(overlaps)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result2 = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
            )

            if result2.fun < 100.0:
                _try_update(result2.x[:N])

    except Exception:
        pass

    # Phase 4: Multi-start with diverse initializations + SLSQP
    try:
        if time.time() < deadline - 15.0:
            np.random.seed(42)
            n_starts = 6
            for trial in range(n_starts):
                if time.time() > deadline - 10.0:
                    break

                if trial == 0:
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / (N/3.0))
                elif trial == 1:
                    h_rand = 0.5 + 0.25 * np.cos(2 * np.pi * np.arange(N) / (N/4.0))
                elif trial == 2:
                    h_rand = np.random.rand(N)
                    kernel = np.ones(7) / 7
                    h_rand = np.convolve(h_rand, kernel, mode='same')
                    h_rand = 0.3 + 0.4 * (h_rand - h_rand.min()) / (h_rand.max() - h_rand.min() + 1e-10)
                elif trial == 3:
                    h_rand = 0.5 + 0.2 * np.sin(2*np.pi*np.arange(N)/8.0) + 0.1 * np.sin(2*np.pi*np.arange(N)/3.0)
                elif trial == 4:
                    # Periodic 3-block pattern
                    period = N // 3
                    h_rand = np.tile([0.3, 0.7, 0.5], N // period + 1)[:N]
                else:
                    h_rand = np.random.rand(N)

                h_rand = np.clip(h_rand, 0.0, 1.0)
                h_rand = _enforce_sum(h_rand)

                def objective_smooth(params):
                    h = params[:N]
                    t = params[N]
                    overlaps = _all_overlaps(h)
                    smax = _smooth_max(overlaps, alpha=200.0)
                    return t + 3000.0 * np.maximum(0.0, smax - t)**2

                x0 = np.concatenate([h_rand, [_max_overlap_full(h_rand) + 0.05]])
                bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

                def constraint_sum(params):
                    return np.sum(params[:N]) - N/2.0

                def constraint_epigraph(params):
                    h = params[:N]
                    t = params[N]
                    overlaps = _all_overlaps(h)
                    return t - np.max(overlaps)

                constraints_ext = [
                    {'type': 'eq', 'fun': constraint_sum},
                    {'type': 'ineq', 'fun': constraint_epigraph}
                ]

                result_ms = minimize(
                    objective_smooth,
                    x0,
                    method='SLSQP',
                    bounds=bounds_ext,
                    constraints=constraints_ext,
                    options={'maxiter': 1500, 'ftol': 1e-14, 'disp': False}
                )

                if result_ms.fun < 100.0:
                    _try_update(result_ms.x[:N])

    except Exception:
        pass

    # Phase 5: Final high-precision polish
    try:
        if time.time() < deadline - 3.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                overlaps = _all_overlaps(h)
                smax = _smooth_max(overlaps, alpha=1000.0)
                return t + 100000.0 * np.maximum(0.0, smax - t)**2

            x0_polish = np.concatenate([h_final, [current_score + 0.001]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                overlaps = _all_overlaps(h)
                return t - np.max(overlaps)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result_final = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 5000, 'ftol': 1e-18, 'disp': False}
            )

            if result_final.fun < 100.0:
                _try_update(result_final.x[:N])

    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
