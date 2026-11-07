"""Erdos minimum overlap: Low-dimensional Fourier parameterization approach."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = 145.0  # Safety margin

def _score(h):
    """Compute the objective value: max correlation / N * 2."""
    if not np.all(np.isfinite(h)):
        return np.inf
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """Project h onto the affine hyperplane sum(h)=N/2 and box [0,1]."""
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _param_to_h(params):
    """
    Convert 6 Fourier parameters to a height sequence.
    params = [a1, a2, a3, b1, b2, b3]
    h[i] = 0.5 + a1*sin(2*pi*i/N + b1) + a2*sin(2*pi*i/N + b2) + a3*sin(2*pi*i/N + b3)
    We use 3 sinusoidal components with independent amplitudes and phases.
    This gives a smooth, low-frequency structure that can be optimized in 6D.
    """
    a1, a2, a3, b1, b2, b3 = params
    x = np.arange(N)
    h = 0.5
    h += a1 * np.sin(2 * np.pi * x / N + b1)
    h += a2 * np.sin(2 * np.pi * x / N + b2)
    h += a3 * np.sin(2 * np.pi * x / N + b3)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h

def _objective_fourier(params):
    """Objective function for Fourier parameter optimization."""
    h = _param_to_h(params)
    return _score(h)

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Define bounds for Fourier parameters
    # Amplitudes: [0, 0.5] to keep h in [0,1] range mostly
    # Phases: [0, 2*pi]
    bounds = [(0, 0.5), (0, 0.5), (0, 0.5), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    # Strategy 1: Differential Evolution on the 6D Fourier parameter space
    # This is a global optimizer that can handle the non-convex landscape
    try:
        result_de = differential_evolution(
            _objective_fourier,
            bounds=bounds,
            seed=42,
            maxiter=200,
            tol=1e-10,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.9,
            polish=False,  # We'll do our own polishing
            update_interval=10,
            callback=lambda xk, convergence: convergence
        )
        
        if result_de.success or result_de.fun < best_score:
            h_de = _param_to_h(result_de.x)
            if _is_valid(h_de):
                s_de = _score(h_de)
                if s_de < best_score:
                    best_score = s_de
                    best_h = h_de.copy()
    except Exception:
        pass
    
    if deadline_check():
        if _is_valid(best_h):
            return best_h
        return np.full(N, 0.5)
    
    # Strategy 2: Local refinement using SLSQP on the Fourier parameters
    # Start from the best Fourier solution found by DE
    if best_score < _score(np.full(N, 0.5)):
        # Find the Fourier params that gave the best score
        # We need to track this, so let's re-run with tracking
        pass
    
    # Let's do a more systematic search:
    # 1. Try many random Fourier parameter sets
    # 2. Refine the best ones with local optimization
    
    rng = np.random.RandomState(12345)
    candidates = []
    
    # Random Fourier parameters
    for _ in range(200):
        params = rng.uniform([0, 0, 0, 0, 0, 0], [0.5, 0.5, 0.5, 2*np.pi, 2*np.pi, 2*np.pi])
        h = _param_to_h(params)
        s = _score(h)
        candidates.append((s, params, h))
    
    # Structured Fourier parameters: low-frequency dominant
    for a1 in [0.2, 0.3, 0.4, 0.45]:
        for a2 in [0, 0.1, 0.2]:
            for a3 in [0, 0.05, 0.1]:
                for b1 in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    for b2 in [0, np.pi]:
                        for b3 in [0, np.pi]:
                            params = [a1, a2, a3, b1, b2, b3]
                            h = _param_to_h(params)
                            s = _score(h)
                            candidates.append((s, params, h))
    
    if deadline_check():
        if _is_valid(best_h):
            return best_h
        return np.full(N, 0.5)
    
    # Sort by score and refine top candidates
    candidates.sort(key=lambda x: x[0])
    
    for s, params, h in candidates[:20]:
        if deadline_check():
            break
        
        # Local refinement with SLSQP on Fourier parameters
        try:
            result_local = minimize(
                _objective_fourier,
                params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result_local.fun < best_score:
                h_local = _param_to_h(result_local.x)
                if _is_valid(h_local):
                    s_local = _score(h_local)
                    if s_local < best_score:
                        best_score = s_local
                        best_h = h_local.copy()
        except Exception:
            pass
    
    if deadline_check():
        if _is_valid(best_h):
            return best_h
        return np.full(N, 0.5)
    
    # Strategy 3: Direct local search on the height sequence itself
    # Using the best Fourier result as starting point
    h = best_h.copy()
    current_score = _score(h)
    
    # Coordinate descent with adaptive step size
    step_size = 0.05
    for iteration in range(500):
        if deadline_check():
            break
        
        improved = False
        for idx in range(N):
            if deadline_check():
                break
            
            # Try moving h[idx] up and down
            for delta in [step_size, -step_size, step_size/2, -step_size/2]:
                h_new = h.copy()
                h_new[idx] += delta
                h_new = np.clip(h_new, 0, 1)
                h_new = _fix_sum(h_new)
                
                if _is_valid(h_new):
                    s_new = _score(h_new)
                    if s_new < current_score - 1e-12:
                        h = h_new
                        current_score = s_new
                        improved = True
                        break
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    if current_score < best_score:
        best_score = current_score
        best_h = h
    
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h
