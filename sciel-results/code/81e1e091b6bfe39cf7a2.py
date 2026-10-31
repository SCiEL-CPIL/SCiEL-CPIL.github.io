"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Approach: Use a Gaussian mixture model (GMM) parameterization to represent
the step function as a sum of Gaussian bumps. This provides a smooth,
low-dimensional parameter space that can be optimized with L-BFGS-B,
escaping local minima inherent in per-element perturbation approaches.
"""

import time
import math
import random

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

N = 600
DELTA = 0.5 / N
DEADLINE = 140.0  # safety margin before 150s kill

_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _midpoints():
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    return x


def _autoconv_peak(h):
    """Compute max of discrete autoconvolution using FFT for speed."""
    h = np.asarray(h, dtype=float)
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    return float(np.max(c))


def _objective_l1(h):
    """max(autoconv) / (sum h)^2 -- the target normalization."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s <= 0:
        return float("inf")
    return _autoconv_peak(h) / (s * s)


def _objective(h):
    return _objective_l1(h)


def _gmm_to_heights(params, n_components=8):
    """Convert GMM parameters to step function heights.
    
    params: array of shape [n_components * 3] = [mu_1, sigma_1, amp_1, mu_2, sigma_2, amp_2, ...]
    Returns: array of N nonnegative heights.
    """
    x = _midpoints()
    h = np.zeros(N)
    for i in range(n_components):
        mu = params[3*i]
        sigma = params[3*i+1]
        amp = params[3*i+2]
        # Clip sigma to avoid numerical issues
        sigma = max(sigma, 0.005)
        h += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return np.clip(h, 0.0, None)


def _objective_from_gmm(params, n_components=8):
    """Objective function for GMM parameters."""
    h = _gmm_to_heights(params, n_components)
    s = float(np.sum(h))
    if s <= 1e-10:
        return 1e10
    return _autoconv_peak(h) / (s * s)


def _gradient_gmm(params, n_components=8):
    """Compute gradient of objective w.r.t. GMM parameters via finite differences."""
    eps = 1e-5
    grad = np.zeros_like(params)
    f0 = _objective_from_gmm(params, n_components)
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        f_p = _objective_from_gmm(params_p, n_components)
        grad[i] = (f_p - f0) / eps
    return grad


def _optimize_gmm(params_init, n_components=8, max_iter=100, time_limit=20.0):
    """Optimize GMM parameters using L-BFGS-B."""
    t_start = time.time()
    
    def obj(p):
        if time.time() - t_start > time_limit:
            raise Exception("timeout")
        return _objective_from_gmm(p, n_components)
    
    try:
        result = minimize(
            obj,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        return result.x, result.fun
    except Exception:
        return params_init, _objective_from_gmm(params_init, n_components)


def _random_gmm_init(n_components=8, seed=None):
    """Generate random GMM parameters."""
    rng = np.random.RandomState(seed)
    params = np.zeros(n_components * 3)
    for i in range(n_components):
        params[3*i] = rng.uniform(-0.2, 0.2)      # mu
        params[3*i+1] = rng.uniform(0.01, 0.15)    # sigma
        params[3*i+2] = rng.uniform(0.5, 3.0)      # amp
    return params


def _local_search(h, max_iters=100, time_limit=15.0):
    """Local search with random perturbations."""
    t_start = time.time()
    h = h.copy()
    best_h = h.copy()
    best_val = _objective(h)
    
    for it in range(max_iters):
        if time.time() - t_start > time_limit or _time_left() <= 10.0:
            break
        
        rng = np.random.RandomState(it * 98765 + 4321)
        h_new = h.copy()
        
        # Perturb a random block
        block_size = rng.randint(10, 50)
        start = rng.randint(0, N - block_size)
        delta = rng.normal(0, 0.05 * np.mean(h), size=block_size)
        h_new[start:start+block_size] += delta
        h_new = np.clip(h_new, 0.0, None)
        
        # Preserve mass
        s_old = float(np.sum(h))
        s_new = float(np.sum(h_new))
        if s_new > 1e-9:
            h_new = h_new * (s_old / s_new)
        
        v_new = _objective(h_new)
        if v_new < best_val - 1e-12:
            best_val = v_new
            best_h = h_new.copy()
            h = h_new
    
    return best_h, best_val


def construct():
    global _t0
    _t0 = time.time()
    
    uniform = np.ones(N)
    best_h = uniform.copy()
    best_val = _objective(uniform)
    
    # --- 1) GMM-based optimization with multiple restarts ---
    n_gmm_restarts = 15
    for seed in range(n_gmm_restarts):
        if _time_left() <= 30.0:
            break
        
        # Random GMM initialization
        params_init = _random_gmm_init(n_components=8, seed=seed * 12345 + 6789)
        
        # Optimize GMM parameters
        params_opt, val_opt = _optimize_gmm(params_init, n_components=8, 
                                             max_iter=80, time_limit=8.0)
        
        # Convert to heights
        h_gmm = _gmm_to_heights(params_opt, n_components=8)
        s = float(np.sum(h_gmm))
        if s > 1e-9:
            h_gmm = h_gmm / s * N  # Normalize
        
        v = _objective(h_gmm)
        if v < best_val - 1e-12:
            best_val = v
            best_h = h_gmm.copy()
    
    # --- 2) Local search polish from best found ---
    if _time_left() > 25.0:
        h_ls, v_ls = _local_search(best_h, max_iters=80, time_limit=15.0)
        if v_ls < best_val - 1e-12:
            best_val = v_ls
            best_h = h_ls.copy()
    
    # --- 3) Additional GMM restarts if time permits ---
    if _time_left() > 20.0:
        for seed in range(15, 25):
            if _time_left() <= 15.0:
                break
            
            params_init = _random_gmm_init(n_components=6, seed=seed * 54321 + 1357)
            params_opt, val_opt = _optimize_gmm(params_init, n_components=6,
                                                 max_iter=60, time_limit=5.0)
            
            h_gmm = _gmm_to_heights(params_opt, n_components=6)
            s = float(np.sum(h_gmm))
            if s > 1e-9:
                h_gmm = h_gmm / s * N
            
            v = _objective(h_gmm)
            if v < best_val - 1e-12:
                best_val = v
                best_h = h_gmm.copy()
    
    # --- 4) Final local search ---
    if _time_left() > 15.0:
        h_final, v_final = _local_search(best_h, max_iters=50, time_limit=10.0)
        if v_final < best_val - 1e-12:
            best_val = v_final
            best_h = h_final.copy()
    
    # --- Safety: never return worse than uniform ---
    if _objective(best_h) > _objective(uniform) + 1e-9:
        best_h = uniform
    
    best_h = np.clip(np.asarray(best_h, dtype=float), 0.0, None)
    return [float(v) for v in best_h]


_t0 = time.time()
