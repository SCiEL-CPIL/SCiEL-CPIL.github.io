import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _soft_max(conv, beta=50.0):
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    sm = max_val + np.log(sum_exp) / beta
    grad_sm = exp_shifted / sum_exp
    return sm, grad_sm

def _ratio(h, beta=50.0):
    """Compute ratio = ||h||_2^2 / (||h||_1 * max(autoconv(h))) for nonnegative h."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    sm, _ = _soft_max(conv, beta)
    if sm < 1e-15:
        return 0.0
    return l2_sq / (l1 * sm)

def _ratio_neg(h, beta=50.0):
    return -_ratio(h, beta)

def _objective_log(log_h, beta=50.0):
    """Objective in log-space: h = exp(log_h), minimize -ratio."""
    h = np.exp(log_h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e6
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    sm, _ = _soft_max(conv, beta)
    if sm < 1e-15:
        return 1e6
    return -l2_sq / (l1 * sm)

def _gaussian_profile(sigma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return h

def _lorentzian_profile(gamma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return h

def _power_profile(p, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    t = np.abs(x - center) / 0.25
    h = np.where(t < 1.0, (1.0 - t) ** p, 0.0)
    return h

def construct():
    start_time = time.time()
    best_heights = np.ones(N)
    best_score = _ratio(best_heights)

    beta = 50.0
    rng = np.random.RandomState(42)

    initializations = []

    for sigma in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        for center in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
            h = _gaussian_profile(sigma, center)
            if np.sum(h) > 1e-10:
                initializations.append(h.copy())

    for gamma in [0.02, 0.03, 0.05, 0.08, 0.1]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            h = _lorentzian_profile(gamma, center)
            if np.sum(h) > 1e-10:
                initializations.append(h.copy())

    for p in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for center in [-0.05, 0.0, 0.05]:
            h = _power_profile(p, center)
            if np.sum(h) > 1e-10:
                initializations.append(h.copy())

    initializations.append(np.ones(N).copy())

    for _ in range(15):
        h = rng.rand(N) ** rng.choice([1, 2, 3, 5])
        if np.sum(h) > 1e-10:
            initializations.append(h.copy())

    for i, h_init in enumerate(initializations):
        if _time_left(start_time) < 15:
            break

        l1 = np.sum(h_init)
        if l1 < 1e-15:
            continue
        h_norm = h_init / l1
        log_h_init = np.log(np.maximum(h_norm, 1e-15))

        try:
            result = minimize(
                _objective_log, log_h_init, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
            )
        except Exception:
            continue

        h_opt = np.exp(result.x)
        l1_opt = np.sum(h_opt)
        if l1_opt < 1e-15:
            continue
        h_opt = h_opt / l1_opt
        score_opt = _ratio(h_opt, beta)

        if score_opt > best_score:
            best_score = score_opt
            best_heights = h_opt.copy()

    if _time_left(start_time) > 20:
        log_h_best = np.log(np.maximum(best_heights, 1e-15))
        for _ in range(5):
            if _time_left(start_time) < 10:
                break
            log_h_pert = log_h_best + 0.1 * rng.randn(N)
            try:
                result = minimize(
                    _objective_log, log_h_pert, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9}
                )
            except Exception:
                continue
            h_opt = np.exp(result.x)
            l1_opt = np.sum(h_opt)
            if l1_opt < 1e-15:
                continue
            h_opt = h_opt / l1_opt
            score_opt = _ratio(h_opt, beta)
            if score_opt > best_score:
                best_score = score_opt
                best_heights = h_opt.copy()

    scale = 1e-8
    final_heights = best_heights * scale
    final_score = _ratio(final_heights, beta)
    if final_score <= 0:
        final_heights = np.ones(N) * 1e-8

    return final_heights.tolist()
