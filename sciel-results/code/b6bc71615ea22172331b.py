"""
Optimized program for the Third Autocorrelation Inequality task.

Structural change: Represent h as the real part of a complex exponential
sequence with linearly varying frequency (chirp), parameterized by initial
phase phi, chirp rate beta, and DC offset c. This 3-parameter family exploits
the quadratic phase to decorrelate autocorrelation sidelobes.

Optimization strategy: Global search via differential evolution on (phi, beta, c)
followed by local L-BFGS-B refinement using a smooth max surrogate for the
non-differentiable max operation.
"""

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
START_TIME = time.time()
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START

def _compute_value(h):
    """Compute the evaluator's exact value."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _chirp_h(phi, beta, c):
    """
    Generate h[i] = cos(phi + beta * i^2) + c.
    The quadratic phase creates destructive interference in autocorrelation
    sidelobes, reducing the peak relative to the sum squared.
    """
    i = np.arange(N, dtype=np.float64)
    return np.cos(phi + beta * i * i) + c

def _smooth_max_obj(params, beta_sm=50.0):
    """
    Objective for optimization: smooth max of autoconvolution / sum^2.
    Uses log-sum-exp as a smooth surrogate for max to enable gradient-based
    optimization.
    """
    phi, beta, c = params
    h = _chirp_h(phi, beta, c)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    # Smooth max via log-sum-exp
    m = np.max(conv)
    sm = m + np.log(np.sum(np.exp((conv - m) * beta_sm))) / beta_sm
    return abs(2.0 * N * sm / (s * s))

def _global_search_chirp(deadline):
    """
    Global search over (phi, beta, c) using differential evolution.
    The chirp rate beta is the key parameter: it controls the rate of
    frequency variation which determines sidelobe suppression.
    """
    if time.time() > deadline:
        return np.ones(N), 2.0
    
    bounds = [
        (0.0, 2.0 * np.pi),   # phi: initial phase
        (-0.02, 0.02),        # beta: chirp rate (small values for N=400)
        (-0.5, 0.5)           # c: DC offset
    ]
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # First do a coarse random search to find promising regions
    for _ in range(200):
        if time.time() > deadline - 30:
            break
        phi = np.random.uniform(0, 2 * np.pi)
        beta = np.random.uniform(-0.02, 0.02)
        c = np.random.uniform(-0.5, 0.5)
        h = _chirp_h(phi, beta, c)
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Then refine with differential evolution from the best region
    try:
        # Use the best found parameters as a starting hint via bounds tightening
        # but DE is global so we just run it
        result = differential_evolution(
            _smooth_max_obj, bounds,
            maxiter=80, popsize=20,
            tol=1e-6, seed=42,
            polish=False,
            init='sobol'
        )
        if time.time() > deadline - 5:
            return best_h, best_val
        if result.fun < np.inf:
            h_de = _chirp_h(*result.x)
            val_de = _compute_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except Exception:
        pass
    
    return best_h, best_val

def _local_refine_chirp(h0, deadline):
    """
    Local refinement using L-BFGS-B on the 3 chirp parameters.
    We extract approximate parameters from h0 and then optimize.
    """
    if time.time() > deadline:
        return h0, _compute_value(h0)
    
    # Estimate initial parameters from h0
    # c ≈ mean(h0), phi ≈ atan2-like estimate, beta from quadratic fit
    c_est = np.mean(h0)
    h_centered = h0 - c_est
    # Fit quadratic phase: h_centered[i] ≈ A * cos(phi + beta * i^2)
    # Use least squares on the phase
    i = np.arange(N, dtype=np.float64)
    # For a cos, the phase is arccos(h_centered / max_abs) but that's noisy
    # Instead, use the fact that for small beta, the phase ≈ phi + beta*i^2
    # We can estimate beta from the second derivative of the phase
    # Simpler: just use the estimate and let L-BFGS-B refine
    phi_est = 0.0
    beta_est = 0.001
    
    def obj(params):
        phi, beta, c = params
        h = _chirp_h(phi, beta, c)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        m = np.max(conv)
        # Use smooth max with moderate temperature
        sm = m + np.log(np.sum(np.exp((conv - m) * 30.0))) / 30.0
        return abs(2.0 * N * sm / (s * s))
    
    x0 = np.array([phi_est, beta_est, c_est])
    
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    try:
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=[(0, 2*np.pi), (-0.02, 0.02), (-0.5, 0.5)],
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if time.time() > deadline - 3:
            return best_h, best_val
        if res.fun < np.inf:
            h_new = _chirp_h(*res.x)
            val_new = _compute_value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    except Exception:
        pass
    
    # Try multiple starting points for robustness
    for trial in range(10):
        if time.time() > deadline - 2:
            break
        phi_try = np.random.uniform(0, 2 * np.pi)
        beta_try = np.random.uniform(-0.015, 0.015)
        c_try = np.random.uniform(-0.4, 0.4)
        x_try = np.array([phi_try, beta_try, c_try])
        try:
            res = minimize(
                obj, x_try, method='L-BFGS-B',
                bounds=[(0, 2*np.pi), (-0.02, 0.02), (-0.5, 0.5)],
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < np.inf:
                h_new = _chirp_h(*res.x)
                val_new = _compute_value(h_new)
                if val_new < best_val:
                    best_val = val_new
                    best_h = h_new.copy()
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    """
    Returns the optimal step heights.
    Uses global search (random + DE) over chirp parameters (phi, beta, c)
    followed by local L-BFGS-B refinement with smooth max surrogate.
    """
    deadline = START_TIME + TIME_LIMIT - 15.0  # Safety margin
    
    # Phase 1: Global search over chirp parameters
    h_global, val_global = _global_search_chirp(deadline - 20)
    
    # Phase 2: Local refinement
    h_final, val_final = _local_refine_chirp(h_global, deadline)
    
    # Final safety check
    if time.time() > deadline - 5:
        return np.ones(N)
    
    final_val = _compute_value(h_final)
    if final_val < 2.0:
        return h_final
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
