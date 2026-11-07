"""Rewritten program for Erdos minimum overlap task."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 100
TIME_LIMIT = 120  # seconds, safety margin before 150s kill

def _value(h):
    """Compute the objective: max cross-correlation of h and 1-h, normalized."""
    return np.max(np.correlate(h, 1 - h, mode="full")) / N * 2

def _validity_penalty(h):
    """Penalty for constraint violations to guide optimizer."""
    # Sum constraint: sum(h) = N/2
    sum_pen = abs(np.sum(h) - N / 2)
    # Bounds: h in [0, 1]
    bound_pen = np.sum(np.maximum(0, h - 1)) + np.sum(np.maximum(0, -h))
    return sum_pen + bound_pen

def _objective(h):
    """Objective with penalties."""
    return _value(h) + 0.1 * _validity_penalty(h)

def _project(h):
    """Project h onto feasible set: sum=N/2, bounds [0,1]."""
    h = np.clip(h, 0, 1)
    # Adjust to meet sum constraint
    current_sum = np.sum(h)
    target_sum = N / 2
    diff = target_sum - current_sum
    if diff > 1e-12:
        # Need to increase sum: add to elements not at upper bound
        mask = h < 1 - 1e-10
        if np.any(mask):
            increment = diff / np.sum(mask)
            h[mask] += increment
            h = np.clip(h, 0, 1)
    elif diff < -1e-12:
        # Need to decrease sum: subtract from elements not at lower bound
        mask = h > 1e-10
        if np.any(mask):
            decrement = -diff / np.sum(mask)
            h[mask] -= decrement
            h = np.clip(h, 0, 1)
    return h

def _random_start():
    """Generate a random feasible starting point."""
    h = np.random.rand(N) * 0.5
    # Scale and project
    h = _project(h * (N / 2) / np.sum(h))
    return h

def _structured_start(kind):
    """Generate structured feasible starting points."""
    if kind == 0:
        # Sine wave
        h = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N * 3)
    elif kind == 1:
        # Block pattern
        h = np.zeros(N)
        h[:N//2] = 0.8
        h[N//2:] = 0.2
    elif kind == 2:
        # Alternating
        h = np.where(np.arange(N) % 2 == 0, 0.8, 0.2)
    elif kind == 3:
        # Triangle wave
        t = np.arange(N) / N
        h = 0.5 + 0.3 * np.abs(2 * (t - 0.5) - 0.5) * 2
    elif kind == 4:
        # Two peaks
        x = np.linspace(0, 1, N)
        h = 0.5 + 0.3 * np.exp(-((x - 0.25) ** 2) / 0.01) - 0.3 * np.exp(-((x - 0.75) ** 2) / 0.01)
    else:
        h = np.random.rand(N) * 0.5
    return _project(h)

def construct():
    """Construct 100 step heights minimizing the overlap value."""
    t0 = time.time()
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    # Generate diverse starting points
    starts = []
    starts.append(np.full(N, 0.5))  # seed
    for k in range(5):
        starts.append(_structured_start(k))
    for _ in range(10):
        starts.append(_random_start())
    
    # Try each starting point with gradient-based optimization
    for i, h0 in enumerate(starts):
        if time.time() - t0 > TIME_LIMIT - 10:
            break
        
        try:
            res = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2,
                             'jac': lambda x: np.ones(N)}],
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_opt = _project(res.x)
            val = _value(h_opt)
            if val < best_val and _validity_penalty(h_opt) < 1e-8:
                best_val = val
                best_h = h_opt
        except Exception:
            continue
    
    # Local search refinement on best solution
    if time.time() - t0 < TIME_LIMIT - 5:
        h_current = best_h.copy()
        val_current = best_val
        for _ in range(2000):
            if time.time() - t0 > TIME_LIMIT - 5:
                break
            idx = np.random.randint(N)
            delta = np.random.choice([-0.05, -0.02, 0.02, 0.05])
            h_trial = h_current.copy()
            h_trial[idx] += delta
            h_trial = _project(h_trial)
            val_trial = _value(h_trial)
            if val_trial < val_current and _validity_penalty(h_trial) < 1e-8:
                h_current = h_trial
                val_current = val_trial
                best_h = h_current
                best_val = val_current
    
    return best_h
