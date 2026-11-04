import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns an array of overlap values C_k for k = 1..N-1 (both positive and negative).
    C_k = sum_i h_i * (1 - h_{i+k})  (with appropriate truncation)
    """
    g = 1.0 - h
    # Full correlation: result[0..2N-2]
    # c[k] = sum_i h[i] * g[i + k - (N-1)]  ... actually np.correlate(h, g, 'full')
    # The center is at index N-1.
    # We want shifts k != 0.
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1
    # c[N-1] is the zero-shift correlation
    # We exclude it.
    # Negative shifts: c[0 : N-1]
    # Positive shifts: c[N : 2N-1]
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a Lagrange multiplier for exact projection.
    """
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    # Bisection on lambda: find lambda such that sum(clip(h - lambda, 0, 1)) = target
    # When lambda < 0, we're adding; when lambda > 0, we're subtracting
    lo, hi = -max(h) - 1, 1 - min(h) + 1
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(h - lam, 0, 1))
    
    for _ in range(60):
        mid = (lo + hi) / 2
        s = sum_at_lambda(mid)
        if abs(s - target) < 1e-12:
            break
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2
    h = np.clip(h - lam, 0, 1)
    
    # Final adjustment for any residual error
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        # Distribute residual to elements not at bounds
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    return h

def _objective_lse(x, beta, lam_sum):
    """
    Objective function using LogSumExp approximation of the max overlap.
    """
    h = np.clip(x, 0, 1)
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    # Avoid numerical issues with large values
    if max_ov > 100:
        # Scale down to avoid overflow in exp
        overlaps = overlaps / 100
        max_ov = max_ov / 100
    
    # Smooth max using LSE
    if beta > 0:
        # Shift for numerical stability
        shifted = overlaps - max_ov
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    # Scale back
    if max_ov > 1:  # We scaled down
        lse *= 100
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit, leave 10s buffer)
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    rng = np.random.default_rng(123)
    
    # Initial best: uniform
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # Sinusoidal
    t = np.arange(N)
    for freq in range(1, 25):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
    
    # Random
    for i in range(80):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
    
    # Evaluate all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Optimization phases with increasing beta
    beta_schedule = [5, 10, 20, 50, 100, 200, 400]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        # Try from best solution
        x0 = best_h.copy()
        
        def obj_func(x):
            return _objective_lse(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Try from random restarts
        num_restarts = 8 if beta <= 50 else 4
        for _ in range(num_restarts):
            if time.time() > deadline - 20.0:
                break
            
            x0 = rng.uniform(0.2, 0.8, size=N)
            x0 = _project_to_feasible(x0)
            
            def obj_rand(x):
                return _objective_lse(x, beta, 100.0)
            
            try:
                result = minimize(
                    obj_rand, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Final polish with high beta
    if time.time() < deadline - 10.0:
        def obj_polish(x):
            return _objective_lse(x, 800.0, 2000.0)
        
        try:
            result = minimize(
                obj_polish, best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Additional local search: coordinate descent on individual elements
    if time.time() < deadline - 5.0:
        improved = True
        max_cd_iterations = 50
        cd_iter = 0
        while improved and cd_iter < max_cd_iterations and time.time() < deadline - 5.0:
            improved = False
            cd_iter += 1
            for i in range(N):
                old_val = best_h[i]
                # Try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    trial = best_h.copy()
                    trial[i] = np.clip(old_val + delta, 0, 1)
                    trial = _project_to_feasible(trial)
                    score = _max_overlap(trial)
                    if score < best_score - 1e-12:
                        best_h = trial
                        best_score = score
                        improved = True
                        break
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    return best_h
