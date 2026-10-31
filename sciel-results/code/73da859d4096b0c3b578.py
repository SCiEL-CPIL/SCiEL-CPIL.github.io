"""Erdos minimum overlap: minimize max overlap of h with shifted (1-h)."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _score(h):
    """Max overlap of h with shifted (1-h). Non-circular shift."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=200.0):
    """Smooth approximation of max via log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=200.0):
    """Gradient of smooth max w.r.t. x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    return exps / np.sum(exps)

def _objective_smooth(params, alpha=200.0):
    """Objective: t + penalty for max-overlap > t. params = [h, t]."""
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t) ** 2
    return t + penalty

def _objective_smooth_grad(params, alpha=200.0):
    """Objective + analytic gradient."""
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t) ** 2
    val = t + penalty

    grad = np.zeros(N + 1)
    # d/dt
    if smax > t:
        grad[N] = 1.0 - 2000.0 * (smax - t)
    else:
        grad[N] = 1.0

    # d(smax)/d(h_j) via smooth-max weights
    w = _smooth_max_grad(overlap, alpha)
    grad_smax_h = np.zeros(N)
    for i in range(N - 1):
        if w[i] > 1e-15:
            grad_smax_h[i] += w[i] * (1.0 - h[i + 1])
            grad_smax_h[i + 1] += w[i] * (-h[i])
    if w[N - 1] > 1e-15:
        grad_smax_h[N - 1] += w[N - 1]

    # Penalty gradient contribution
    if smax > t:
        dp = 2000.0 * (smax - t)
        grad[:N] += dp * grad_smax_h
        grad[N] -= dp
    else:
        grad[:N] += 0.0  # no penalty gradient

    return val, grad

def _project_to_feasible(h):
    """Project h to satisfy sum = N/2 and 0 <= h <= 1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        return np.full(N, 0.5)
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        idx = int(np.argmin(h)) if diff > 0 else int(np.argmax(h))
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Analytic 2-block initial guess: h = A on first half, B on second half."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A                        # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _polish(h_start, deadline):
    """Local polish using L-BFGS-B with smooth-max objective."""
    h_start = _project_to_feasible(h_start.copy())
    current_score = _score(h_start)

    # Build initial params [h, t]
    t_init = current_score + 0.01
    x0 = np.concatenate([h_start, [t_init]])
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

    try:
        result = minimize(
            _objective_smooth,
            x0,
            jac=_objective_smooth_grad,
            method='L-BFGS-B',
            bounds=bounds_ext,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        h_opt = _project_to_feasible(result.x[:N])
        score_opt = _score(h_opt)
        if score_opt < current_score - 1e-12:
            return h_opt, score_opt
    except Exception:
        pass

    return h_start, current_score

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Start with analytic 2-block solution
    h_final = _two_block_init()
    current_score = _score(h_final)

    # Phase 1: Global search via Differential Evolution
    try:
        if time.time() < deadline - 30.0:
            def de_objective(h_params):
                h = _project_to_feasible(h_params)
                return _score(h)

            bounds = [(0.0, 1.0)] * N
            result_de = minimize(
                de_objective,
                h_final,
                method='DifferentialEvolution',
                bounds=bounds,
                options={
                    'maxiter': 80,
                    'popsize': 18,
                    'tol': 1e-10,
                    'seed': 42,
                    'polish': False,
                }
            )
            if result_de.fun < current_score - 1e-12:
                h_de = _project_to_feasible(result_de.x)
                score_de = _score(h_de)
                if score_de < current_score:
                    h_final = h_de
                    current_score = score_de
    except Exception:
        pass

    # Phase 2: Local polish from best DE result (or initial)
    if time.time() < deadline - 10.0:
        h_final, current_score = _polish(h_final, deadline)

    # Phase 3: Additional polish with different alpha for robustness
    try:
        if time.time() < deadline - 5.0:
            h_start = h_final.copy()
            t_init = current_score + 0.01
            x0 = np.concatenate([h_start, [t_init]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            def obj_a500(params):
                return _objective_smooth(params, alpha=500.0)

            def grad_a500(params):
                return _objective_smooth_grad(params, alpha=500.0)

            result2 = minimize(
                obj_a500,
                x0,
                jac=grad_a500,
                method='L-BFGS-B',
                bounds=bounds_ext,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt2 = _project_to_feasible(result2.x[:N])
            score_opt2 = _score(h_opt2)
            if score_opt2 < current_score - 1e-12:
                h_final = h_opt2
                current_score = score_opt2
    except Exception:
        pass

    # Final projection and exact sum correction
    h_final = _project_to_feasible(h_final)
    diff = N / 2.0 - h_final.sum()
    if abs(diff) > 1e-15:
        idx = int(np.argmin(h_final)) if diff > 0 else int(np.argmax(h_final))
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)

    return h_final
