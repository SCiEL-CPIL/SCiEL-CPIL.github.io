"""
Low-dimensional parametric curve projection for Erdos minimum overlap.

Strategy: Model h[i] as a sum of a few sinusoids (plus a constant offset)
to keep the mean at 0.5, and optimize the low-dimensional parameters.
This reduces the search space from 100 dimensions to ~10, making global
optimization feasible.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
DEADLINE = 145.0

def _score(h):
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
    """
    Adjust h to sum to N/2 while keeping values in [0, 1].
    """
    target = N / 2.0
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    
    for _ in range(50):
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

def _param_to_h(params, num_freqs=3):
    """
    Convert parameters to height sequence.
    params: [offset, a1, p1, a2, p2, a3, p3]
    offset is fixed at 0.5 for sum constraint, but we can allow slight variation
    and then fix sum. Actually, to strictly satisfy sum=50, we must have mean=0.5.
    A sinusoid has mean 0, so h = 0.5 + sum(ai * cos(wi * x + pi)) has mean 0.5.
    We just need to ensure h is in [0, 1].
    """
    x = np.arange(N)
    h = np.full(N, 0.5)
    
    for i in range(num_freqs):
        a = params[i * 2]
        p = params[i * 2 + 1]
        # Frequency is determined by phase? No, let's use frequency directly or integer periods.
        # Using continuous frequency can lead to non-periodic behavior in the discrete window.
        # Better to use integer periods to ensure clean periodicity.
        # But differential evolution works on continuous spaces.
        # Let's interpret p as the period (1.0 to 50.0).
        if p < 1e-6:
            continue
        h += a * np.cos(2 * np.pi * x / p)
        
    # h should have mean 0.5.
    # Clip to [0, 1] and fix sum.
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h

def _objective(params, num_freqs=3):
    h = _param_to_h(params, num_freqs)
    return _score(h)

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE

    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Strategy:
    # 1. Use Differential Evolution to find good parameters for 1, 2, and 3 sinusoids.
    # 2. Refine with local optimization (Nelder-Mead or L-BFGS-B) on the parameters.
    # 3. Also try a final local refinement on the heights themselves (coordinate descent).
    
    num_freqs_list = [1, 2, 3]
    
    for num_freqs in num_freqs_list:
        if deadline_check():
            break
            
        # Define bounds for parameters
        # Amplitudes: [0, 0.5] to stay within [0, 1] roughly
        # Periods: [1, 50]
        bounds = []
        for i in range(num_freqs):
            bounds.append((0.0, 0.5))  # Amplitude
            bounds.append((1.0, 50.0)) # Period
            
        # Differential Evolution
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(num_freqs,),
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                init='sobol',
                polish=True
            )
            
            if result.success or result.fun < best_score:
                best_params = result.x
                best_score = result.fun
                best_h = _param_to_h(best_params, num_freqs)
                
        except Exception:
            pass
        
        if deadline_check():
            break
            
        # Local refinement on parameters
        if best_score < _score(np.full(N, 0.5)):
            try:
                # Use Nelder-Mead for local refinement of parameters
                res_nm = minimize(
                    _objective,
                    best_params,
                    args=(num_freqs,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_nm.fun < best_score:
                    best_score = res_nm.fun
                    best_h = _param_to_h(res_nm.x, num_freqs)
            except Exception:
                pass

    # Final local refinement on heights (coordinate descent)
    # This helps escape any local minima in the parametric space that might be fixable by small height tweaks.
    if not deadline_check():
        h = best_h.copy()
        current_score = _score(h)
        rng = np.random.RandomState(42)
        
        # Coordinate descent with random restarts
        for _ in range(200):
            if deadline_check():
                break
            idx = rng.randint(N)
            # Try small perturbations
            for delta in [-0.05, -0.01, 0.01, 0.05]:
                h_new = h.copy()
                h_new[idx] += delta
                h_new = np.clip(h_new, 0.0, 1.0)
                h_new = _fix_sum(h_new)
                if not _is_valid(h_new):
                    continue
                s_new = _score(h_new)
                if s_new < current_score:
                    current_score = s_new
                    h = h_new
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_h = h

    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
        
    return best_h
