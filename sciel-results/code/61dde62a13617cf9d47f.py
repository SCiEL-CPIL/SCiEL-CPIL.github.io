"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _overlap_matrix(h):
    """
    Compute the overlap matrix M where M[i,j] = h[i] * (1 - h[j]).
    The score is max over all i != j (shifts) of sum_k M[k, (k+shift)%N].
    But the task says 'maximum overlap of h with shifted 1-h'.
    
    Re-reading: "minimize the maximum overlap of h with shifted 1-h"
    Overlap for shift s: sum_i h[i] * (1 - h[(i+s) % N])
    
    This is what the original code computes. Let's stick with that definition.
    """
    pass

def _score(h):
    """
    Compute the maximum overlap over all shifts.
    overlap[s] = sum_i h[i] * (1 - h[(i+s) % N])
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Use FFT for circular autocorrelation
    H = np.fft.fft(h)
    # r[s] = sum_i h[i] * h[(i+s) % n]  (circular autocorrelation)
    # Note: np.fft.ifft(np.abs(H)**2) gives r[s] = sum_i h[(i+s)%n] * h[i]
    r = np.real(np.fft.ifft(np.abs(H)**2))
    total = np.sum(h)
    # overlap[s] = sum_i h[i] - sum_i h[i]*h[(i+s)%n] = total - r[s]
    overlaps = total - r
    # We consider shifts s = 1, 2, ..., n-1 (non-trivial shifts)
    # Shift 0 gives sum h[i](1-h[i]) which is not the "shifted" overlap usually meant
    # But the problem says "shifted 1-h", implying s != 0.
    # Let's check the seed code: it uses shifts = np.arange(N), which includes 0.
    # However, typically "shifted" implies a non-zero shift.
    # Let's look at the inspiration scores. They are around 1.15.
    # The seed score is 1.0.
    # If we include shift 0, the overlap is sum h(1-h), which for h=0.5 is N*0.25 = 25.
    # If we exclude shift 0, we look at other lags.
    
    # The seed code computes:
    # ov = _overlap_values(h, np.arange(N))
    # which includes shift 0.
    # Let's assume the definition includes all shifts 0..N-1.
    
    return float(np.max(overlaps))

def _smooth_max(h, beta):
    """
    Smooth maximum of the overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    H = np.fft.fft(h)
    r = np.real(np.fft.ifft(np.abs(H)**2))
    total = np.sum(h)
    overlaps = total - r
    
    m = np.max(overlaps)
    # Avoid overflow in exp
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    # This is a simple projection; for better results, we might need iterative correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Add diff to the element with the most room
        # Or just add to all again and clip, which might oscillate.
        # A better way: add diff to elements that are not at bounds.
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # All at bounds, which means sum is either 0 or N, impossible for N/2
            pass
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        # Find an element that can absorb the difference
        if diff > 0:
            # Add to element with smallest value (most room up)
            idx = np.argmin(h)
        else:
            # Subtract from element with largest value (most room down)
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _logistic_h(x):
    """
    Generate step heights from logistic map parameters.
    x = [r, x0, offset, scale]
    r in [3.5, 4.0], x0 in [0, 1], offset in [-0.5, 0.5], scale in [0.5, 1.5]
    """
    r = x[0]
    x0 = x[1]
    offset = x[2]
    scale = x[3]
    
    h = np.zeros(N)
    x_curr = x0
    for i in range(N):
        h[i] = x_curr
        x_curr = r * x_curr * (1.0 - x_curr)
    
    # Transform to [0, 1] with mean 0.5
    # h is in [0, 1] already for r in [0, 4] and x0 in [0, 1]
    # Apply offset and scale
    h = scale * (h - 0.5) + 0.5 + offset
    
    return h

def _logistic_objective(x):
    """
    Objective for logistic map parameter optimization.
    Penalize if sum is not N/2 or if values out of [0, 1].
    """
    h = _logistic_h(x)
    
    # Check bounds
    if np.any(h < 0) or np.any(h > 1):
        # Penalty for out of bounds
        penalty = np.sum(np.maximum(0, h - 1)**2) + np.sum(np.maximum(0, -h)**2)
        return 1e6 + penalty
    
    # Check sum
    sum_h = np.sum(h)
    sum_penalty = (sum_h - N/2.0)**2
    
    # Smooth max overlap
    score = _smooth_max(h, 50.0)
    
    return score + 100.0 * sum_penalty

def _gaussian_objective(x):
    """
    Objective for direct optimization of h.
    """
    h = np.clip(x, 0.0, 1.0)
    sum_h = np.sum(h)
    sum_penalty = (sum_h - N/2.0)**2
    score = _smooth_max(h, 50.0)
    return score + 100.0 * sum_penalty

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Try logistic map parameters with global search
    # Parameters: r, x0, offset, scale
    # r in [3.5, 4.0], x0 in [0.01, 0.99], offset in [-0.2, 0.2], scale in [0.8, 1.2]
    
    # Use differential evolution or basin hopping
    # Since we have a deadline, use a limited number of iterations
    
    if time.time() - t0 < _DEADLINE - 30:
        try:
            bounds_log = [(3.5, 4.0), (0.01, 0.99), (-0.2, 0.2), (0.8, 1.2)]
            # Use scipy.optimize.differential_evolution
            from scipy.optimize import differential_evolution
            res = differential_evolution(_logistic_objective, bounds_log, 
                                          maxiter=50, popsize=10, 
                                          seed=42, tol=1e-6)
            if res.fun < best_score:
                h_log = _logistic_h(res.x)
                _update_best(h_log)
        except Exception:
            pass
    
    # 2. Try direct optimization with SLSQP from multiple starts
    if time.time() - t0 < _DEADLINE - 20:
        # Start from logistic result if available, else random
        if best_h is not None:
            starts = [best_h.copy()]
        else:
            starts = []
        
        # Add some random starts
        for _ in range(5):
            h_rand = np.random.uniform(0, 1, N)
            starts.append(h_rand)
        
        # Add a sinusoidal start
        i = np.arange(N)
        h_sin = 0.5 + 0.3 * np.sin(2 * np.pi * i / N)
        starts.append(h_sin)
        
        for h0 in starts:
            if time.time() - t0 > _DEADLINE - 10:
                break
            
            h0 = _fix_sum(h0)
            
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            try:
                res = minimize(_gaussian_objective, h0.copy(), method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 200, 'ftol': 1e-9})
                if res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass
    
    # 3. Final polish with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        try:
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            # Use a higher beta for smoother max approximation
            def _objective_high_beta(x):
                h = np.clip(x, 0.0, 1.0)
                return _smooth_max(h, 100.0) + 100.0 * (np.sum(h) - N/2.0)**2
            
            res = minimize(_objective_high_beta, best_h.copy(), method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass
    
    if best_h is None:
        # Fallback: uniform 0.5
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
# EVOLVE-BLOCK-END
