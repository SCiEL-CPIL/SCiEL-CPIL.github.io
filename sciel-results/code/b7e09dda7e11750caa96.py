"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_overlap(h, shift):
    """Compute overlap integral of h(x) with h(x+shift) for step functions on [0,2]."""
    # h is defined on [0, 2] with N equal-width steps, each of width 2/N
    # h(x) = h[i] for x in [2*i/N, 2*(i+1)/N)
    # We need integral_0^2 h(x) * h(x+shift) dx where h is extended periodically or zero outside [0,2]
    # For the problem, we consider h on [0,2] and shifts s in [0,2]
    # The overlap is integral over x where both x and x+s are in [0,2]
    
    dw = 2.0 / N
    shift_idx = int(round(shift / dw))
    if shift_idx >= N:
        return 0.0
    
    # For each step i of h(x), it overlaps with step i+shift_idx of h(x+shift)
    # if that step exists
    valid = min(N - shift_idx, N)
    overlap = 0.0
    for i in range(valid):
        # Both h[i] and h[i+shift_idx] are defined
        overlap += h[i] * h[i + shift_idx] * dw
    
    return overlap


def _compute_max_overlap(h):
    """Find maximum overlap of h with shifted (1-h)."""
    dw = 2.0 / N
    max_overlap = 0.0
    
    # Check all possible shifts
    for shift_idx in range(N):
        shift = shift_idx * dw
        # overlap of h(x) with (1-h(x+shift))
        # = integral h(x) * (1 - h(x+shift)) dx
        # = integral h(x) dx - integral h(x)*h(x+shift) dx
        # = 1 - integral h(x)*h(x+shift) dx
        
        valid = min(N - shift_idx, N)
        overlap_hh = 0.0
        for i in range(valid):
            overlap_hh += h[i] * h[i + shift_idx] * dw
        
        overlap = 1.0 - overlap_hh
        if overlap > max_overlap:
            max_overlap = overlap
    
    return max_overlap


def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    
    def objective(h):
        return _compute_max_overlap(h)
    
    def objective_grad(h):
        # Numerical gradient for the max overlap
        eps = 1e-6
        grad = np.zeros(N)
        f0 = objective(h)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            f_plus = objective(h_plus)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Use scipy minimize with bounds
    bounds = [(0.0, 1.0)] * N
    
    best_h = h0.copy()
    best_score = objective(h0)
    
    # Try SLSQP with constraints
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
    
    try:
        result = minimize(
            objective,
            h0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.fun < best_score:
            best_h = result.x
            best_score = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return best_h
    
    # Try L-BFGS-B without constraints, then project
    try:
        result2 = minimize(
            objective,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
        )
        
        if result2.fun < best_score:
            best_h = result2.x
            best_score = result2.fun
    except Exception:
        pass
    
    # Ensure sum constraint is satisfied
    # Adjust to make sum exactly N/2
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-10:
        diff = N/2.0 - current_sum
        # Distribute the difference evenly, clipping to [0,1]
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Re-adjust to ensure exact sum
        current_sum = np.sum(best_h)
        if abs(current_sum - N/2.0) > 1e-10:
            diff = N/2.0 - current_sum
            best_h[0] += diff
            best_h[0] = np.clip(best_h[0], 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
