"""
Strategy: Represent step heights as the real part of a complex exponential with
cubic phase modulation (Fresnel chirp). This creates a signal whose instantaneous
frequency changes quadratically, potentially decorrelating sidelobes more
effectively than linear chirps. We optimize the cubic coefficient, initial phase,
amplitude, and offset using differential evolution, followed by local refinement.

The key insight is that cubic phase modulation (h[n] = offset + amp * cos(phase + beta * (n/N)^3))
produces a signal with a quadratic frequency sweep, which may distribute autocorrelation
energy more evenly across lags compared to constant or linearly-varying frequency signals.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400
_DEADLINE = 140.0  # Wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _chirp_signal(params):
    """
    Generate a chirp signal from parameters.
    params = [offset, amplitude, phase, cubic_coeff]
    h[n] = offset + amplitude * cos(phase + cubic_coeff * (n/N)^3)
    """
    offset, amp, phase, cubic = params
    n = np.arange(N, dtype=np.float64)
    t = n / N
    h = offset + amp * np.cos(phase + cubic * t**3)
    return h


def _objective(params):
    """Objective function for differential evolution: minimize the task value."""
    h = _chirp_signal(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    return val


def _local_refine(h, deadline):
    """
    Local refinement using coordinate-wise greedy search.
    Try small perturbations to individual elements and keep improvements.
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng(999)
    
    # Try different step sizes
    step_sizes = [0.5, 0.1, 0.05, 0.01, 0.005]
    
    for step_size in step_sizes:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try increasing
                h_test = best_h.copy()
                h_test[i] += step_size
                v_test = _evaluate(h_test)
                if np.isfinite(v_test) and v_test < best_val - 1e-12:
                    best_val = v_test
                    best_h = h_test
                    improved = True
                # Try decreasing
                h_test = best_h.copy()
                h_test[i] -= step_size
                v_test = _evaluate(h_test)
                if np.isfinite(v_test) and v_test < best_val - 1e-12:
                    best_val = v_test
                    best_h = h_test
                    improved = True
    
    return best_h


def _random_search(deadline):
    """
    Random search over the chirp parameter space with multiple restarts.
    """
    best_val = np.inf
    best_params = None
    
    rng = np.random.default_rng(123)
    
    # Define bounds for parameters
    # offset: [0, 3], amplitude: [0.1, 2], phase: [0, 2*pi], cubic: [-20, 20]
    bounds = [(0, 3), (0.1, 2), (0, 2 * np.pi), (-20, 20)]
    
    n_evals = 0
    max_evals = 5000
    
    while time.time() < deadline and n_evals < max_evals:
        # Sample random parameters
        params = np.array([
            rng.uniform(*bounds[0]),
            rng.uniform(*bounds[1]),
            rng.uniform(*bounds[2]),
            rng.uniform(*bounds[3])
        ])
        
        val = _objective(params)
        n_evals += 1
        
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_params = params.copy()
    
    return best_params, best_val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    Uses cubic chirp representation with differential evolution and local refinement.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Random search to find good initial chirp parameters
    if time.time() < deadline - 30:
        params_init, val_init = _random_search(deadline - 30)
        if params_init is not None and val_init < best_val:
            best_val = val_init
            best_h = _chirp_signal(params_init)
    
    # Phase 2: Differential evolution to optimize chirp parameters
    if time.time() < deadline - 20:
        bounds = [(0, 3), (0.1, 2), (0, 2 * np.pi), (-20, 20)]
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True,
                options={'maxfev': 2000, 'init': 'sobol'}
            )
            
            if result.success and np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_h = _chirp_signal(result.x)
            elif hasattr(result, 'x') and np.all(np.isfinite(result.x)):
                val_de = _objective(result.x)
                if np.isfinite(val_de) and val_de < best_val:
                    best_val = val_de
                    best_h = _chirp_signal(result.x)
        except Exception as e:
            pass
    
    # Phase 3: Local refinement of the best candidate
    if time.time() < deadline - 5:
        refined_h = _local_refine(best_h, deadline - 5)
        refined_val = _evaluate(refined_h)
        if np.isfinite(refined_val) and refined_val < best_val:
            best_val = refined_val
            best_h = refined_h
    
    # Phase 4: Try additional chirp variants with different structures
    if time.time() < deadline - 10:
        rng = np.random.default_rng(777)
        
        # Try quadratic chirp: cos(phase + beta * t^2)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            offset = rng.uniform(0, 3)
            amp = rng.uniform(0.1, 2)
            phase = rng.uniform(0, 2 * np.pi)
            quad_coeff = rng.uniform(-10, 10)
            
            n = np.arange(N, dtype=np.float64)
            t = n / N
            h_test = offset + amp * np.cos(phase + quad_coeff * t**2)
            
            val_test = _evaluate(h_test)
            if np.isfinite(val_test) and val_test < best_val:
                best_val = val_test
                best_h = h_test.copy()
        
        # Try mixed chirp: cos(phase + beta1 * t + beta2 * t^2 + beta3 * t^3)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            offset = rng.uniform(0, 3)
            amp = rng.uniform(0.1, 2)
            phase = rng.uniform(0, 2 * np.pi)
            b1 = rng.uniform(-5, 5)
            b2 = rng.uniform(-10, 10)
            b3 = rng.uniform(-20, 20)
            
            n = np.arange(N, dtype=np.float64)
            t = n / N
            h_test = offset + amp * np.cos(phase + b1 * t + b2 * t**2 + b3 * t**3)
            
            val_test = _evaluate(h_test)
            if np.isfinite(val_test) and val_test < best_val:
                best_val = val_test
                best_h = h_test.copy()
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        # Fall back to seed if optimization failed
        best_h = np.ones(N)
    
    return best_h
