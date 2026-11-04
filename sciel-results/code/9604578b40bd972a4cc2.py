import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0

_DEADLINE = 150.0
_SAFETY = 15.0
_start = 0.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _forward(params):
    """
    Forward pass: params -> h (nonnegative heights).
    params layout: [centers(5), sigmas(5), amps(5), extra(5)] = 20 params
    h[i] = sum_k amps[k] * exp(-0.5*((i-centers[k])/sigmas[k])^2)
    """
    centers = params[:5]
    sigmas = np.abs(params[5:10]) + 0.5
    amps = np.abs(params[10:15]) + 0.01
    extra = params[15:20]
    
    i = np.arange(N, dtype=float)
    h = np.zeros(N)
    for k in range(5):
        h += amps[k] * np.exp(-0.5 * ((i - centers[k]) / sigmas[k]) ** 2)
    
    # Add small baseline from extra params to allow more flexibility
    for k in range(5):
        h += 0.1 * np.abs(extra[k]) * np.cos(2 * np.pi * extra[k] * i / N)
    
    h = np.maximum(h, 0.0)
    return h


def _ratio_from_h(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return 0.0
    l2_sq = float(np.sum(h**2))
    f = np.convolve(h, h, mode='full')
    f_inf = float(np.max(f))
    if f_inf <= 1e-300:
        return 0.0
    return l2_sq / (l1 * f_inf)


def _log_ratio_and_grad(params):
    """
    Compute log(ratio) and its gradient w.r.t. params using finite differences
    for the soft-max part, analytic for the rest.
    Returns (log_ratio, grad_log_ratio).
    """
    h = _forward(params)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return -1e10, np.zeros_like(params)
    
    l2_sq = float(np.sum(h**2))
    
    # Autoconvolution
    f = np.convolve(h, h, mode='full')
    f_inf = float(np.max(f))
    if f_inf <= 1e-300:
        return -1e10, np.zeros_like(params)
    
    # Use soft-max for differentiability
    temp = 0.05
    f_shifted = f - f_inf
    f_inf_soft = f_inf + temp * np.log(np.sum(np.exp(f_shifted / temp)))
    
    ratio = l2_sq / (l1 * f_inf_soft)
    if ratio <= 0:
        return -1e10, np.zeros_like(params)
    
    log_r = np.log(ratio)
    
    # Gradient via central finite differences (20 params, cheap enough)
    grad = np.zeros_like(params)
    eps = 1e-5
    for j in range(len(params)):
        p_plus = params.copy()
        p_plus[j] += eps
        p_minus = params.copy()
        p_minus[j] -= eps
        
        h_plus = _forward(p_plus)
        l1_plus = float(np.sum(h_plus))
        l2_plus = float(np.sum(h_plus**2))
        f_plus = np.convolve(h_plus, h_plus, mode='full')
        f_inf_plus = float(np.max(f_plus))
        f_inf_plus_soft = f_inf_plus + temp * np.log(np.sum(np.exp((f_plus - f_inf_plus) / temp)))
        r_plus = l2_plus / (l1_plus * f_inf_plus_soft) if l1_plus > 1e-300 and f_inf_plus_soft > 1e-300 else 1e-300
        log_r_plus = np.log(max(r_plus, 1e-300))
        
        h_minus = _forward(p_minus)
        l1_minus = float(np.sum(h_minus))
        l2_minus = float(np.sum(h_minus**2))
        f_minus = np.convolve(h_minus, h_minus, mode='full')
        f_inf_minus = float(np.max(f_minus))
        f_inf_minus_soft = f_inf_minus + temp * np.log(np.sum(np.exp((f_minus - f_inf_minus) / temp)))
        r_minus = l2_minus / (l1_minus * f_inf_minus_soft) if l1_minus > 1e-300 and f_inf_minus_soft > 1e-300 else 1e-300
        log_r_minus = np.log(max(r_minus, 1e-300))
        
        grad[j] = (log_r_plus - log_r_minus) / (2 * eps)
    
    return log_r, grad


def _neg_log_ratio(params):
    log_r, _ = _log_ratio_and_grad(params)
    return -log_r


def _neg_log_ratio_jac(params):
    log_r, grad = _log_ratio_and_grad(params)
    return -log_r, -grad


def _polish(h, time_budget=5.0):
    """Local polish using coordinate descent with finer steps."""
    start_t = time.time()
    for _ in range(500):
        if time.time() - start_t > time_budget:
            break
        improved = False
        cur_ratio = _ratio_from_h(h)
        for i in range(N):
            if time.time() - start_t > time_budget:
                break
            # Try several perturbations
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                h_test = h.copy()
                h_test[i] = max(0.0, h[i] + delta)
                r_test = _ratio_from_h(h_test)
                if r_test > cur_ratio + 1e-12:
                    h = h_test
                    cur_ratio = r_test
                    improved = True
                    break
        if not improved:
            break
    return h, _ratio_from_h(h)


def construct():
    global _start
    _start = time.time()
    
    best_h = np.ones(N)
    best_ratio = _ratio_from_h(best_h)
    
    # Phase 1: Multi-start gradient-based optimization with Gaussian bump parameterization
    n_restarts = 25
    
    for restart in range(n_restarts):
        if _time_left() < 10:
            break
        
        # Generate diverse initial parameters
        if restart == 0:
            # Structured: concentrated center
            centers = np.array([25.0, 24.0, 26.0, 23.0, 27.0])
            sigmas = np.array([2.0, 3.0, 3.0, 4.0, 4.0])
            amps = np.array([1.0, 0.5, 0.5, 0.3, 0.3])
            extra = np.zeros(5)
        elif restart == 1:
            # Two peaks
            centers = np.array([15.0, 35.0, 25.0, 10.0, 40.0])
            sigmas = np.array([3.0, 3.0, 5.0, 2.0, 2.0])
            amps = np.array([1.0, 1.0, 0.3, 0.2, 0.2])
            extra = np.zeros(5)
        elif restart == 2:
            # Broad single peak
            centers = np.array([25.0, 25.0, 25.0, 25.0, 25.0])
            sigmas = np.array([8.0, 10.0, 6.0, 12.0, 5.0])
            amps = np.array([1.0, 0.5, 0.5, 0.3, 0.5])
            extra = np.zeros(5)
        else:
            centers = np.random.uniform(5, 45, 5)
            sigmas = np.random.uniform(1.0, 10.0, 5)
            amps = np.random.uniform(0.1, 2.0, 5)
            extra = np.random.uniform(-2, 2, 5)
        
        params = np.concatenate([centers, sigmas, amps, extra])
        
        # Optimize
        try:
            result = minimize(
                _neg_log_ratio,
                params,
                jac=_neg_log_ratio_jac,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 30}
            )
            
            if result.fun < 1e10:
                h_opt = _forward(result.x)
                h_opt = np.maximum(h_opt, 0.0)
                
                if np.sum(h_opt) > 1e-10:
                    ratio = _ratio_from_h(h_opt)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_h = h_opt.copy()
        except Exception:
            pass
        
        # Also try without gradient (in case gradient is noisy)
        if _time_left() < 5:
            break
        try:
            result2 = minimize(
                _neg_log_ratio,
                params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result2.fun < 1e10:
                h_opt2 = _forward(result2.x)
                h_opt2 = np.maximum(h_opt2, 0.0)
                if np.sum(h_opt2) > 1e-10:
                    ratio2 = _ratio_from_h(h_opt2)
                    if ratio2 > best_ratio:
                        best_ratio = ratio2
                        best_h = h_opt2.copy()
        except Exception:
            pass
    
    # Phase 2: Polish the best solution with coordinate descent
    if _time_left() > 8:
        time_budget = max(1.0, _time_left() - 8)
        best_h, best_ratio = _polish(best_h, time_budget=time_budget)
    
    # Phase 3: Try a few more random starts if time remains
    if _time_left() > 10:
        for _ in range(5):
            if _time_left() < 5:
                break
            centers = np.random.uniform(5, 45, 5)
            sigmas = np.random.uniform(1.0, 8.0, 5)
            amps = np.random.uniform(0.1, 1.5, 5)
            extra = np.random.uniform(-1.5, 1.5, 5)
            params = np.concatenate([centers, sigmas, amps, extra])
            try:
                result = minimize(
                    _neg_log_ratio,
                    params,
                    jac=_neg_log_ratio_jac,
                    method='L-BFGS-B',
                    options={'maxiter': 40, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                if result.fun < 1e10:
                    h_opt = _forward(result.x)
                    h_opt = np.maximum(h_opt, 0.0)
                    if np.sum(h_opt) > 1e-10:
                        ratio = _ratio_from_h(h_opt)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_h = h_opt.copy()
            except Exception:
                pass
    
    return list(best_h)
