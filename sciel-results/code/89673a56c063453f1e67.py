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
    """Enforce sum constraint and bounds precisely."""
    if target_sum is None:
        target_sum = N / 2.0
    h = h.copy()
    for _ in range(10):
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

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

    # First: standard SLSQP on the raw max-overlap objective
    try:
        def objective_raw(h):
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return float(np.max(overlap))

        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
        bounds = [(0.0, 1.0)] * N

        result = minimize(
            objective_raw,
            h0.copy(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
        )

        if result.fun < current_score:
            h_opt = _enforce_constraints(result.x.copy())
            new_score = _score(h_opt)
            if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                h_final = h_opt.copy()
                current_score = new_score

    except Exception:
        pass

    # Second: smooth surrogate with epigraph variable t
    try:
        if time.time() < deadline - 10.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=200.0)
                return t + 1000.0 * np.maximum(0.0, smax - t)**2

            x0 = np.concatenate([h_final, [current_score + 0.1]])
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

            result2 = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )

            if result2.fun < 100.0:
                h_opt2 = _enforce_constraints(result2.x[:N].copy())
                new_score2 = _score(h_opt2)
                if new_score2 < current_score and np.all(h_opt2 >= 0.0) and np.all(h_opt2 <= 1.0):
                    h_final = h_opt2.copy()
                    current_score = new_score2

    except Exception:
        pass

    # Third: high-precision polish using the smooth objective with tighter alpha
    try:
        if time.time() < deadline - 5.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 10000.0 * np.maximum(0.0, smax - t)**2

            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
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

            result3 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
            )

            if result3.fun < 100.0:
                h_opt3 = _enforce_constraints(result3.x[:N].copy())
                new_score3 = _score(h_opt3)
                if new_score3 < current_score and np.all(h_opt3 >= 0.0) and np.all(h_opt3 <= 1.0):
                    h_final = h_opt3.copy()
                    current_score = new_score3

    except Exception:
        pass

    # Fourth: systematic multi-start with diverse structured initializations
    try:
        if time.time() < deadline - 15.0:
            np.random.seed(42)
            initializations = []
            
            # Golden ratio blocks
            A = (3.0 - math.sqrt(5.0)) / 2.0
            B = 1.0 - A
            h_golden = np.zeros(N)
            h_golden[:N//2] = A
            h_golden[N//2:] = B
            initializations.append(h_golden)
            
            # Alternating high/low
            h_alt = np.zeros(N)
            h_alt[::2] = 0.7
            h_alt[1::2] = 0.3
            initializations.append(h_alt)
            
            # Sine wave
            h_sin = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)
            initializations.append(h_sin)
            
            # Cosine wave
            h_cos = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)
            initializations.append(h_cos)
            
            # Random with structure
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                h_rand = np.random.rand(N)
                h_rand = 0.5 * h_rand + 0.5 * 0.5
                initializations.append(h_rand)
            
            # Run SLSQP on each initialization
            for h_init in initializations:
                if time.time() > deadline - 10.0:
                    break
                
                h_init = _enforce_constraints(h_init.copy())
                
                def objective_raw(h):
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    return float(np.max(overlap))

                constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
                bounds = [(0.0, 1.0)] * N

                result_r = minimize(
                    objective_raw,
                    h_init.copy(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
                )

                if result_r.fun < current_score:
                    h_opt_r = _enforce_constraints(result_r.x.copy())
                    new_score_r = _score(h_opt_r)
                    if new_score_r < current_score and np.all(h_opt_r >= 0.0) and np.all(h_opt_r <= 1.0):
                        h_final = h_opt_r.copy()
                        current_score = new_score_r

    except Exception:
        pass

    # Fifth: Final high-precision polish with very tight tolerance
    try:
        if time.time() < deadline - 3.0:
            def objective_final(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=1000.0)
                return t + 50000.0 * np.maximum(0.0, smax - t)**2

            x0_final = np.concatenate([h_final, [current_score + 0.001]])
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

            result_final = minimize(
                objective_final,
                x0_final,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-18, 'disp': False}
            )

            if result_final.fun < 100.0:
                h_opt_final = _enforce_constraints(result_final.x[:N].copy())
                new_score_final = _score(h_opt_final)
                if new_score_final < current_score and np.all(h_opt_final >= 0.0) and np.all(h_opt_final <= 1.0):
                    h_final = h_opt_final.copy()
                    current_score = new_score_final

    except Exception:
        pass

    return h_final
