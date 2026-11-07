import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _correlate_full(h, g):
    """Full cross-correlation of h and g, length 2N-1."""
    return np.correlate(h, g, mode='full')

def _compute_value(h):
    """The actual evaluator value."""
    corr = _correlate_full(h, 1.0 - h)
    return np.max(corr) / N * 2

def _objective(h):
    """Objective to minimize: max correlation value."""
    corr = _correlate_full(h, 1.0 - h)
    return np.max(corr)

def _objective_grad(h):
    """Return (max_corr, gradient) for the max-correlation objective."""
    h = np.asarray(h)
    g = 1.0 - h
    corr = _correlate_full(h, g)
    max_idx = np.argmax(corr)
    max_val = corr[max_idx]
    
    # For lag k in 'full' mode (0..2N-2), let lag = k - (N-1)
    # corr[k] = sum_j h[j] * g[j - lag] where g[j-lag] is valid
    # d(corr[k])/dh[i] = g[i-lag] - h[i+lag] (with bounds checking)
    lag = max_idx - (N - 1)
    grad = np.zeros(N)
    
    for i in range(N):
        j1 = i - lag
        if 0 <= j1 < N:
            grad[i] += g[j1]
        j2 = i + lag
        if 0 <= j2 < N:
            grad[i] += -h[j2]
    
    return max_val, grad

def _smooth_objective(h, temperature=1.0):
    """Log-sum-exp smooth approximation of max correlation."""
    corr = _correlate_full(h, 1.0 - h)
    m = np.max(corr)
    log_sum_exp = m + temperature * np.log(np.sum(np.exp((corr - m) / temperature)))
    return log_sum_exp

def _smooth_objective_grad(h, temperature=1.0):
    """Gradient of the smooth max correlation objective."""
    h = np.asarray(h)
    g = 1.0 - h
    corr = _correlate_full(h, g)
    
    m = np.max(corr)
    weights = np.exp((corr - m) / temperature)
    weights /= np.sum(weights)
    
    grad = np.zeros(N)
    for k in range(len(corr)):
        if weights[k] < 1e-15:
            continue
        lag = k - (N - 1)
        w = weights[k]
        for i in range(N):
            j1 = i - lag
            if 0 <= j1 < N:
                grad[i] += w * g[j1]
            j2 = i + lag
            if 0 <= j2 < N:
                grad[i] += -w * h[j2]
    
    return np.sum(weights * corr), grad

def _project_to_feasible(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(500):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-14:
            break
        # Only adjust elements not at bounds
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.sum(mask) == 0:
            break
        step = diff / np.sum(mask)
        h[mask] += step
        h = np.clip(h, 0.0, 1.0)
    # Final exact sum correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute among all non-bound elements
        mask = (h > 1e-15) & (h < 1.0 - 1e-15)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def _gradient_descent_smooth(h0, temperature=0.5, max_iter=3000, lr_init=0.01, lr_decay=0.999, time_limit=None):
    """Gradient descent with smooth objective and projection."""
    h = h0.copy()
    lr = lr_init
    start_time = time.time()
    best_h = h.copy()
    best_val = _objective(h)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        val, grad = _smooth_objective_grad(h, temperature)
        
        # Gradient step
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        
        new_val = _objective(h_new)
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
        
        h = h_new
        lr *= lr_decay
    
    return best_h, best_val

def _gradient_descent_exact(h0, max_iter=5000, lr_init=0.005, lr_decay=0.9995, time_limit=None):
    """Gradient descent with exact max objective and projection."""
    h = h0.copy()
    lr = lr_init
    start_time = time.time()
    best_h = h.copy()
    best_val = _objective(h)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        val, grad = _objective_grad(h)
        
        # Gradient step
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        
        new_val = _objective(h_new)
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
        
        h = h_new
        lr *= lr_decay
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [np.full(N, 0.5)]
    
    # Random perturbations
    for _ in range(8):
        h_pert = np.full(N, 0.5) + rng.randn(N) * 0.15
        starts.append(_project_to_feasible(h_pert))
    
    # Sine waves with different frequencies and phases
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in [0, 0.5, 1.0]:
            h_sine = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase * np.pi)
            starts.append(_project_to_feasible(h_sine))
    
    # Linear ramp
    h_ramp = np.linspace(0.0, 1.0, N)
    starts.append(_project_to_feasible(h_ramp))
    
    # Two-level: first 50 are 1, rest 0
    h_bin = np.zeros(N)
    h_bin[:50] = 1.0
    starts.append(h_bin.copy())
    
    # Alternating patterns
    for base in [0.2, 0.25, 0.3, 0.35]:
        h_alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
        starts.append(_project_to_feasible(h_alt))
    
    # Three-level patterns
    for pattern in [[0.0, 0.5, 1.0], [0.0, 0.4, 0.8], [0.1, 0.5, 0.9]]:
        h_three = np.tile(pattern, N // len(pattern) + 1)[:N]
        starts.append(_project_to_feasible(h_three))
    
    # Spread pattern
    h_spread = np.full(N, 0.5)
    for i in range(0, N, 2):
        h_spread[i] = 0.75
        if i + 1 < N:
            h_spread[i+1] = 0.25
    starts.append(_project_to_feasible(h_spread))
    
    # Cosine patterns
    for freq in [1, 2, 3]:
        h_cos = 0.5 + 0.45 * np.cos(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_cos))
    
    def try_slsqp(h0, max_iter=800, ftol=1e-10):
        """Run SLSQP optimization and return result if improved."""
        nonlocal best_h, best_val
        try:
            res = minimize(
                _objective,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
            )
            h_res = _project_to_feasible(res.x)
            val = _objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
                return True
        except Exception:
            pass
        return False
    
    def try_smooth_slsqp(h0, temperature=0.5, max_iter=500, ftol=1e-10):
        """Run SLSQP with smooth objective."""
        nonlocal best_h, best_val
        try:
            res = minimize(
                lambda x: _smooth_objective(x, temperature),
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
            )
            h_res = _project_to_feasible(res.x)
            val = _objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
                return True
        except Exception:
            pass
        return False
    
    # Phase 1: SLSQP on all starts
    for h0 in starts:
        if time.time() > deadline - 40:
            break
        try_slsqp(h0, max_iter=600, ftol=1e-10)
    
    # Phase 2: Smooth gradient descent on best solution
    if time.time() < deadline - 30:
        time_left = deadline - 30 - time.time()
        h_gd, val_gd = _gradient_descent_smooth(best_h.copy(), temperature=0.5, max_iter=5000, lr_init=0.01, time_limit=time_left)
        if val_gd < best_val:
            best_val = val_gd
            best_h = h_gd.copy()
    
    # Phase 3: Exact gradient descent refinement
    if time.time() < deadline - 20:
        time_left = deadline - 20 - time.time()
        h_gd2, val_gd2 = _gradient_descent_exact(best_h.copy(), max_iter=8000, lr_init=0.003, lr_decay=0.9998, time_limit=time_left)
        if val_gd2 < best_val:
            best_val = val_gd2
            best_h = h_gd2.copy()
    
    # Phase 4: High-precision SLSQP polish
    if time.time() < deadline - 10:
        try_slsqp(best_h.copy(), max_iter=2000, ftol=1e-14)
    
    # Phase 5: Another smooth pass with lower temperature
    if time.time() < deadline - 5:
        time_left = deadline - 5 - time.time()
        h_gd3, val_gd3 = _gradient_descent_smooth(best_h.copy(), temperature=0.1, max_iter=3000, lr_init=0.005, time_limit=time_left)
        if val_gd3 < best_val:
            best_val = val_gd3
            best_h = h_gd3.copy()
    
    # Final projection
    best_h = _project_to_feasible(best_h)
    
    return best_h
