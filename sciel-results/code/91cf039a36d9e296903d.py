"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _local_search(h, max_iter=200, step_init=0.1, step_decay=0.95, time_limit=None, start_time=None):
    """Local search with coordinate descent and shrinking step size."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 120.0
    
    h = np.array(h, dtype=np.float64).copy()
    current_val = _compute_value(h)
    step = step_init
    best_h = h.copy()
    best_val = current_val
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Try increasing
            h[i] += step
            if h[i] < 0:
                h[i] = 0
            val = _compute_value(h)
            if val > current_val + 1e-12:
                current_val = val
                improved = True
                if val > best_val:
                    best_val = val
                    best_h = h.copy()
            else:
                h[i] -= step
            
            # Try decreasing
            h[i] -= step
            if h[i] < 0:
                h[i] = 0
            val = _compute_value(h)
            if val > current_val + 1e-12:
                current_val = val
                improved = True
                if val > best_val:
                    best_val = val
                    best_h = h.copy()
            else:
                h[i] += step
        
        if not improved:
            step *= step_decay
            if step < 1e-6:
                break
    
    return best_h, best_val

def _hump_init():
    """Create a hump-shaped initial guess."""
    x = np.linspace(-1, 1, N)
    # Triangular hump
    h = np.maximum(0, 1 - np.abs(x))
    h = h / np.max(h)
    return h

def _random_restart():
    """Generate a random initial guess."""
    return np.random.rand(N) * 0.5

def construct():
    """Construct 50 nonnegative step heights to maximize the objective value."""
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit
    
    best_h = None
    best_val = -1.0
    
    # Start with hump initialization
    h = _hump_init()
    h, val = _local_search(h, max_iter=300, step_init=0.1, time_limit=time_limit, start_time=start_time)
    if val > best_val:
        best_val = val
        best_h = h.copy()
    
    # Try several random restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() - start_time > time_limit - 10:
            break
        
        h = _random_restart()
        # Quick local search with fewer iterations for exploration
        h, val = _local_search(h, max_iter=100, step_init=0.05, time_limit=time_limit, start_time=start_time)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    # Final refinement on best solution
    if best_h is not None:
        best_h, best_val = _local_search(best_h, max_iter=500, step_init=0.01, time_limit=time_limit, start_time=start_time)
    
    if best_h is None:
        return [1.0] * N
    
    return list(best_h)

# EVOLVE-BLOCK-END
