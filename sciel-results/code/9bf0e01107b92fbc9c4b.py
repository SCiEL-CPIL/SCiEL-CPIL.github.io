"""Program for Erdos minimum overlap task using low-dimensional parametric projection."""
import numpy as np
from scipy.optimize import differential_evolution

N = 100


def _evaluate_h(h):
    """Compute the objective value for a given height sequence."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return float('inf')
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return float('inf')
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_and_clip(h):
    """Project h onto sum=N/2 and clip to [0,1], then re-project."""
    h = h.copy()
    # First clip
    h = np.clip(h, 0.0, 1.0)
    # Project onto sum = N/2
    h = h + (N / 2.0 - np.sum(h)) / N
    # Clip again
    h = np.clip(h, 0.0, 1.0)
    # Final projection
    h = h + (N / 2.0 - np.sum(h)) / N
    return h


def construct():
    """
    Use a sum of K sinusoids with variable frequencies, amplitudes, and phases.
    Use differential evolution to minimize the objective.
    """
    i = np.arange(N)
    
    # Number of sinusoidal components
    K = 5
    
    # Define bounds for each parameter
    # For each sinusoid k: freq, amp, phase
    # freq: 0.01 to 0.5 (cycles per index)
    # amp: 0.0 to 0.45
    # phase: 0.0 to 2*pi
    
    bounds = []
    for _ in range(K):
        bounds.append((0.01, 0.5))  # frequency
        bounds.append((0.0, 0.45))  # amplitude
        bounds.append((0.0, 2 * np.pi)) # phase
    
    def objective(params):
        h_raw = np.full(N, 0.5)
        for k in range(K):
            f = params[3*k]
            a = params[3*k + 1]
            p = params[3*k + 2]
            h_raw += a * np.sin(2 * np.pi * f * i + p)
        
        h = _project_and_clip(h_raw)
        return _evaluate_h(h)
    
    # Run differential evolution
    # maxiter and popsize control the search effort
    # 120s limit allows for a decent search
    # Using seed for reproducibility
    try:
        result = differential_evolution(objective, bounds, maxiter=2000, popsize=30, seed=42, tol=1e-7, polish=True)
    except Exception:
        # Fallback to a simple random search if DE fails or times out (unlikely with these settings)
        best_val = float('inf')
        best_params = None
        for _ in range(1000):
            params = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            val = objective(params)
            if val < best_val:
                best_val = val
                best_params = params
        if best_params is None:
            return np.full(N, 0.5)
        best_params = best_params
    else:
        best_params = result.x
    
    best_h_raw = np.full(N, 0.5)
    for k in range(K):
        f = best_params[3*k]
        a = best_params[3*k + 1]
        p = best_params[3*k + 2]
        best_h_raw += a * np.sin(2 * np.pi * f * i + p)
        
    best_h = _project_and_clip(best_h_raw)
    
    # Final verification
    assert np.all(np.isfinite(best_h)), "Heights must be finite"
    assert np.all(best_h >= -1e-9) and np.all(best_h <= 1 + 1e-9), "Heights out of bounds"
    assert abs(np.sum(best_h) - N / 2) <= 1e-9 * N, "Sum constraint violated"
    
    return best_h
