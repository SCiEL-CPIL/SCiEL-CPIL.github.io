"""Rewritten program: optimize h to minimize max autocorrelation with (1-h)."""
import time
import numpy as np

N = 100
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0  # seconds to leave before kill

def _compute_value(h):
    """Compute the objective value: max(np.correlate(h, 1-h, mode='full')) / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _is_valid(h):
    """Check if h satisfies all constraints."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1.0 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True

def _project_to_feasible(h):
    """Project h onto feasible region: sum(h)=N/2, 0<=h<=1."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to meet sum constraint
    target = N / 2.0
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute correction evenly, then clip and iterate
        for _ in range(100):
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
            diff = target - np.sum(h)
            if abs(diff) < 1e-12:
                break
    return h

def _gradient_descent(h0, lr=0.01, max_iter=5000, tol=1e-10):
    """
    Gradient descent with projection to minimize max correlation.
    Uses a smooth approximation of max via log-sum-exp for gradient.
    """
    h = h0.copy()
    best_h = h.copy()
    best_val = _compute_value(h)
    
    # Soft-max parameter for smooth max approximation
    # We'll use a temperature-based approach
    T = 0.01  # small temperature for sharp max approximation
    
    for it in range(max_iter):
        # Compute correlation
        corr = np.correlate(h, 1.0 - h, mode="full")
        
        # Smooth max via log-sum-exp
        # corr has length 2*N-1
        max_corr = np.max(corr)
        
        # Gradient of smooth max w.r.t. corr
        # softmax weights
        shifted = corr - max_corr
        weights = np.exp(shifted / T)
        weights /= np.sum(weights)
        
        # Gradient of corr[k] w.r.t. h[i]
        # corr[k] = sum_j h[j] * (1-h[j+k-N+1]) for appropriate indices
        # For full correlation: corr[k] = sum_{j} h[j] * (1-h[j-(k-(N-1))])
        # Let's compute gradient numerically for reliability
        
        # Actually, let's use finite differences for the gradient of the smooth max
        # This is expensive but N=100 so it's manageable
        eps = 1e-7
        grad = np.zeros(N)
        
        # For each h[i], compute how it affects corr
        # corr[k] = sum_j h[j] * (1 - h[j - k + N - 1]) where index is valid
        # d corr[k] / d h[i] = (1 - h[i - k + N - 1]) + h[i] * (-1) * (delta if i - k + N - 1 valid)
        # More carefully:
        # corr[k] = sum_{j: 0<=j<N, 0<=j-(k-N+1)<N} h[j] * (1 - h[j - (k - N + 1)])
        # d corr[k] / d h[i] = (1 - h[i - (k - N + 1)]) if 0 <= i-(k-N+1) < N
        #                     + h[i] * (-1) if 0 <= i < N and the other term uses h[i]
        #                     i.e., if j - (k-N+1) = i, i.e., j = i + k - N + 1, and 0<=j<N
        #                     so h[i] * (-1) * 1 if 0 <= i + k - N + 1 < N
        
        for i in range(N):
            dci = 0.0
            for k in range(2 * N - 1):
                offset = k - (N - 1)  # so corr[k] involves h[j] and h[j - offset]
                # Term 1: h[i] * (1 - h[i - offset]) if 0 <= i - offset < N
                idx1 = i - offset
                if 0 <= idx1 < N:
                    dci += weights[k] * (1.0 - h[idx1])
                # Term 2: h[j] * (1 - h[i]) where j - offset = i, so j = i + offset
                idx2 = i + offset
                if 0 <= idx2 < N:
                    dci += weights[k] * (-1.0) * h[idx2]
            grad[i] = dci
        
        # Gradient of value = max(corr)/N*2
        # d value / d h[i] = grad[i] / N * 2
        grad_value = grad / N * 2.0
        
        # Take gradient step
        h_new = h - lr * grad_value
        
        # Project to feasible region
        h_new = _project_to_feasible(h_new)
        
        val = _compute_value(h_new)
        if val < best_val - tol:
            best_val = val
            best_h = h_new.copy()
            h = h_new
            # Adaptive learning rate: decrease if we made progress
            # lr *= 0.999
        else:
            # No improvement, reduce learning rate
            lr *= 0.95
            if lr < 1e-10:
                break
    
    return best_h, best_val

def _sinusoidal_start(freq, phase):
    """Generate a sinusoidal starting point."""
    t = np.linspace(0, 2 * np.pi * freq, N)
    h = 0.5 + 0.3 * np.sin(t + phase)
    return _project_to_feasible(h)

def construct():
    """Construct 100 step heights minimizing the correlation objective."""
    start_time = time.time()
    
    # Initial seed
    h_seed = np.full(N, 0.5)
    best_h = h_seed.copy()
    best_val = _compute_value(h_seed)  # Should be 0.5
    
    # Try multiple sinusoidal starting points with different frequencies and phases
    start_configs = []
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            start_configs.append((freq, phase))
    
    # Also add some random starts
    rng = np.random.RandomState(42)
    for _ in range(20):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        start_configs.append(('random', h_rand))
    
    for config in start_configs:
        # Check time
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        if isinstance(config[0], str) and config[0] == 'random':
            h0 = config[1]
        else:
            freq, phase = config
            h0 = _sinusoidal_start(freq, phase)
        
        # Run gradient descent
        h_opt, val_opt = _gradient_descent(h0, lr=0.05, max_iter=3000)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
    
    # Final polish: run a few more iterations with smaller learning rate
    if time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        h_polish, val_polish = _gradient_descent(best_h, lr=0.001, max_iter=2000)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    
    return best_h
