"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _enforce_sum(h, target_sum=None):
    """Enforce sum(h) = target_sum (default N/2) and 0<=h<=1 with iterative projection."""
    if target_sum is None:
        target_sum = len(h) / 2.0
    h = h.copy()
    n = len(h)
    for _ in range(200):
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) < 1e-12:
            break
        # Only adjust non-bound elements to avoid wasting projection effort
        # But simple uniform shift is robust.
        h = h + diff / n
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment for non-clipped elements to ensure exact sum
    mask = (h > 1e-10) & (h < 1.0 - 1e-10)
    if np.any(mask):
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) > 1e-12:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    # If still not exact (due to clipping), distribute remainder to any available space
    s = np.sum(h)
    diff = target_sum - s
    if abs(diff) > 1e-12:
        # Find indices with room
        if diff > 0:
            room = 1.0 - h
        else:
            room = h
        indices = np.argsort(-room)
        i = 0
        remaining = diff
        while abs(remaining) > 1e-12 and i < n:
            idx = indices[i]
            step = min(abs(remaining), room[idx])
            if remaining > 0:
                h[idx] += step
            else:
                h[idx] -= step
            remaining -= step
            i += 1
    return h

def _generate_logistic_h(x0, r):
    """
    Generate h from logistic map x_{n+1} = r * x_n * (1 - x_n).
    Parameters:
    x0: initial condition in (0, 1)
    r: control parameter in [0, 4]
    """
    x = x0
    seq = np.zeros(N)
    for i in range(N):
        seq[i] = x
        x = r * x * (1.0 - x)
        # Ensure x stays in [0,1] to avoid numerical issues if r is slightly off or precision loss
        if x < 0: x = 0.0
        elif x > 1: x = 1.0
    # Normalize/Scale seq to [0,1] if needed, but logistic map outputs are already in [0,1] for r in [0,4] and x0 in [0,1]
    # However, the mean of the sequence depends on r and x0.
    # We need sum(h) = N/2, i.e., mean = 0.5.
    # We will project the sequence to satisfy the sum constraint.
    h = _enforce_sum(seq, target_sum=N/2.0)
    return h

def _chaotic_objective(params):
    """
    Objective function for chaotic parameter search.
    params: [x0, r]
    """
    x0, r = params
    # Ensure valid ranges
    if x0 <= 0 or x0 >= 1:
        return 10.0
    if r < 0 or r > 4:
        return 10.0
    
    h = _generate_logistic_h(x0, r)
    
    # Check validity
    if not (np.all(h >= -1e-10) and np.all(h <= 1.0 + 1e-10)):
        return 10.0
    s = np.sum(h)
    if abs(s - N/2.0) > 1e-8:
        return 10.0
        
    return _score(h)

def _local_polish(h_init, deadline):
    """
    Perform local optimization on the full vector using SLSQP.
    """
    def objective_full(h):
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        # Use smooth max to help gradient
        alpha = 200.0
        x_max = np.max(overlap)
        return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (overlap - x_max))))

    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
    bounds_full = [(0.0, 1.0)] * N

    try:
        result = minimize(
            objective_full,
            h_init,
            method='SLSQP',
            bounds=bounds_full,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
        )
        return result.x
    except Exception:
        return h_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial best: Constant 0.5
    # overlap = 0.5 * (1 - 0.5) = 0.25
    h_best = np.full(N, 0.5)
    best_score = _score(h_best)
    
    # Try a simple 2-block solution as another baseline
    # A = (3 - sqrt(5))/2 approx 0.382, B = 1-A approx 0.618
    # Overlap at boundary: A*(1-B) = A*A = 0.146, B*(1-A) = B*B = 0.382. Max is 0.382. Worse than 0.25.
    # Wait, the overlap is h_i * (1 - h_{i+1}).
    # If h is constant 0.5, overlap is 0.25 everywhere.
    # If we alternate 0 and 1? Sum = N/2.
    # h = [1, 0, 1, 0...].
    # i=0: 1*(1-0)=1. Bad.
    # h = [0, 1, 0, 1...].
    # i=0: 0*(1-1)=0.
    # i=1: 1*(1-0)=1. Bad.
    # So constant 0.5 is a decent baseline (score 0.25).

    # Strategy: Search over logistic map parameters (x0, r)
    # r in [0, 4], x0 in [0.01, 0.99]
    bounds = [(0.01, 0.99), (0.0, 4.0)]
    
    best_params = None
    
    # Run Differential Evolution
    try:
        if time.time() < deadline - 30.0:
            # Use a small population and maxiter to fit in time
            # We can also run multiple shorter DEs if needed, but one should suffice for now
            de_result = differential_evolution(
                _chaotic_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=10,
                tol=1e-6,
                init='latinhypercube',
                polish=False # We will polish manually
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
                h_best = _generate_logistic_h(best_params[0], best_params[1])
                
    except Exception as e:
        print(f"DE failed: {e}")

    # If we found a good chaotic sequence, polish it locally
    if best_params is not None and time.time() < deadline - 10.0:
        h_polished = _local_polish(h_best, deadline)
        h_polished = _enforce_sum(h_polished, target_sum=N/2.0)
        sc = _score(h_polished)
        if sc < best_score:
            best_score = sc
            h_best = h_polished

    # Additional random restarts with chaotic maps if time permits
    if time.time() < deadline - 5.0:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            x0 = np.random.uniform(0.01, 0.99)
            r = np.random.uniform(0.0, 4.0)
            h_cand = _generate_logistic_h(x0, r)
            sc = _score(h_cand)
            if sc < best_score:
                best_score = sc
                h_best = h_cand
                # Polish this new best
                if time.time() < deadline - 5.0:
                    h_polished = _local_polish(h_best, deadline)
                    h_polished = _enforce_sum(h_polished, target_sum=N/2.0)
                    sc2 = _score(h_polished)
                    if sc2 < best_score:
                        best_score = sc2
                        h_best = h_polished

    # Final check and return
    h_best = _enforce_sum(h_best, target_sum=N/2.0)
    return h_best
