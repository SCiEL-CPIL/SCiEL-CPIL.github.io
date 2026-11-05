"""Rewritten program for the third autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 125.0  # safety margin before 150s kill


def _value_from_h(h):
    """Compute the task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))


def _smooth_objective_and_grad(h, beta=50.0):
    """
    Smooth surrogate with analytic gradient.
    Objective: |2N * LSE(conv(h,h)) / sum(h)^2|
    where LSE is log-sum-exp approximation to max.
    
    Gradient derivation:
    conv[j] = sum_i h[i]*h[j-i], so d(conv[j])/d(h[k]) = 2*h[j-k] if 0<=j-k<N else 0.
    dLSE/dh[k] = sum_j w[j] * d(conv[j])/d(h[k]) = 2 * correlate(w, h, 'valid')[k]
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = conv - m
    
    # Numerical stability
    max_shifted = np.max(shifted)
    if max_shifted > 0 and beta * max_shifted > 500:
        beta_eff = 500.0 / max_shifted
    else:
        beta_eff = beta
    
    exp_shifted = np.exp(beta_eff * shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    LSE = m + np.log(sum_exp) / beta_eff
    
    val = 2.0 * N * LSE / (s * s)
    obj = abs(val)
    
    # dLSE_dh[k] = 2 * sum_j weights[j] * h[j-k] = 2 * np.correlate(weights, h, 'valid')[k]
    dLSE_dh = 2.0 * np.correlate(weights, h, mode='valid')
    
    # d(val)/d(h_k) = 2N * (dLSE_dh[k]/s^2 - 2*LSE/s^3)
    sign_val = 1.0 if val >= 0 else -1.0
    dval_dh = 2.0 * N * (dLSE_dh / (s * s) - 2.0 * LSE / (s ** 3))
    grad = sign_val * dval_dh
    
    return obj, grad


def _spectral_init(rng, n_modes=10):
    """
    Generate initial h with flat power spectrum and suppressed DC.
    This encourages the autoconvolution peak to be small.
    """
    freqs = rng.integers(1, N // 2, size=n_modes)
    phases = rng.uniform(0, 2 * np.pi, size=n_modes)
    amplitudes = rng.uniform(0.5, 1.5, size=n_modes)
    
    h = np.zeros(N)
    for f, ph, amp in zip(freqs, phases, amplitudes):
        h += amp * np.cos(2 * np.pi * f * np.arange(N) / N + ph)
    
    # Subtract mean to control DC
    h -= np.mean(h)
    
    # Small perturbation for diversity
    h += rng.normal(0, 0.05, N)
    
    return h


def _local_refine(h_init, t0, deadline):
    """Local refinement with increasing beta for progressively sharper max approximation."""
    best_h = h_init.copy()
    best_val = _value_from_h(best_h)
    
    for beta in [5.0, 20.0, 50.0, 150.0, 500.0]:
        if time.time() - t0 > deadline - 3:
            break
        try:
            res = minimize(
                _smooth_objective_and_grad,
                best_h,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            hard_val = _value_from_h(res.x)
            if hard_val < best_val:
                best_h = res.x.copy()
                best_val = hard_val
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Start with seed
    best_h = np.ones(N).copy()
    best_val = _value_from_h(best_h)
    
    # Multi-start with spectral initialization
    n_starts = 8
    for i in range(n_starts):
        if time.time() - t0 > deadline - 20:
            break
        
        h_init = _spectral_init(rng, n_modes=8 + i)
        h_refined, val = _local_refine(h_init, t0, deadline)
        
        if val < best_val:
            best_h = h_refined
            best_val = val
    
    # Final high-precision polish
    if time.time() - t0 < deadline - 3:
        for beta in [1000.0, 2000.0]:
            if time.time() - t0 > deadline - 1:
                break
            try:
                res = minimize(
                    _smooth_objective_and_grad,
                    best_h,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                hard_val = _value_from_h(res.x)
                if hard_val < best_val:
                    best_h = res.x.copy()
                    best_val = hard_val
            except Exception:
                pass
    
    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-15:
        return np.ones(N)
    
    return best_h


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
