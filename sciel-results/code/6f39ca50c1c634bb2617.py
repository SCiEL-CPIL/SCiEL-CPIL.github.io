"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s wall-clock kill


def _compute_score(x):
    """Compute max|autoconv| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    autoconv = np.correlate(x, x, mode='full')
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (s * s)


def _autoconv(x):
    """Compute full autoconvolution."""
    return np.correlate(x, x, mode='full')


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2.
    
    Uses local search to optimize real-valued vector starting from all-ones.
    The all-ones vector achieves 1/N, but real-valued optimization may improve.
    """
    x = np.ones(N)
    best_score = _compute_score(x)
    best_x = x.copy()
    
    # Local search: try adjusting individual elements
    step_size = 0.1
    np.random.seed(42)
    
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5:
        return best_x
    
    # Gradient-free local search with adaptive step size
    for iteration in range(5000):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - 5:
            break
        
        # Pick a random index to modify
        i = np.random.randint(N)
        old_val = x[i]
        
        # Try several perturbations
        best_local_val = old_val
        best_local_score = best_score
        
        for delta in [-step_size, -step_size/2, step_size/2, step_size]:
            x[i] = old_val + delta
            sc = _compute_score(x)
            if sc < best_local_score:
                best_local_score = sc
                best_local_val = old_val + delta
        
        x[i] = best_local_val
        if best_local_score < best_score:
            best_score = best_local_score
            best_x = x.copy()
            # Increase step size slightly on success
            step_size = min(step_size * 1.2, 1.0)
        else:
            # Decrease step size on failure
            step_size = max(step_size * 0.8, 0.01)
    
    # Final polish: try smaller steps
    step_size = 0.01
    for iteration in range(3000):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - 3:
            break
        
        i = np.random.randint(N)
        old_val = x[i]
        best_local_val = old_val
        best_local_score = _compute_score(x)
        
        for delta in [-step_size, step_size]:
            x[i] = old_val + delta
            sc = _compute_score(x)
            if sc < best_local_score:
                best_local_score = sc
                best_local_val = old_val + delta
        
        x[i] = best_local_val
        if best_local_score < best_score:
            best_score = best_local_score
            best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
