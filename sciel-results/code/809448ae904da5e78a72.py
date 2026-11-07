"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _objective(h):
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    return np.max(corr)

def _smooth_objective(h, beta):
    """Smooth approximation of max(corr) using log-sum-exp."""
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    max_corr = np.max(corr)
    shifted = corr - max_corr
    return max_corr + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_grad(h, beta):
    """Analytical gradient of the smooth objective using log-sum-exp.
    
    LSE = max(corr) + log(sum(exp(beta*(corr - max(corr)))))/beta
    d LSE / d h = sum_k w_k * d corr[k] / d h
    where w_k = exp(beta*(corr[k]-max)) / sum(exp(beta*(corr-max)))
    
    For corr = correlate(h, 1-h, 'full'), corr[k] = sum_j h[j]*(1-h[j+m]) where m=k-(N-1).
    d corr[k]/d h[i] = (1-h[i+m]) - h[i-m] with validity checks.
    """
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    max_corr = np.max(corr)
    shifted = corr - max_corr
    exp_shifted = np.exp(beta * shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Compute gradient analytically using vectorized operations
    grad = np.zeros(N)
    for k in range(2 * N - 1):
        if weights[k] < 1e-15:
            continue
        m = k - (N - 1)
        # d corr[k] / d h[i] = (1 - h[i+m]) if 0<=i+m<N else 0, minus h[i-m] if 0<=i-m<N else 0
        dk_dh = np.zeros(N)
        
        # First term: (1 - h[i+m]) for valid i+m
        if m >= 0:
            # i+m in [0, N) => i in [0, N-m)
            dk_dh[:N-m] += (1.0 - h[m:])
        else:
            # i+m in [0, N) => i in [-m, N)
            dk_dh[-m:] += (1.0 - h[:N+m])
        
        # Second term: -h[i-m] for valid i-m
        if m >= 0:
            # i-m in [0, N) => i in [m, N)
            dk_dh[m:] -= h[:N-m]
        else:
            # i-m in [0, N) => i in [0, N+m)
            dk_dh[:N+m] -= h[-m:]
        
        grad += weights[k] * dk_dh
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    def try_update(h_candidate, best_state):
        if not np.all(np.isfinite(h_candidate)):
            return best_state
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate = h_candidate + diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
        val = _objective(h_candidate)
        if val < best_state[1]:
            return (h_candidate.copy(), val)
        return best_state
    
    # Seed
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    best_state = (best_h, best_val)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]
    
    # Generate diverse initializations - tuned for speed
    inits = []
    
    # Cosine-based inits with more frequencies
    for k in range(1, 15):
        for amp in [0.2, 0.3, 0.4, 0.45, 0.49]:
            h = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            inits.append(h)
    
    # Multi-frequency combinations
    for k1 in [1, 2, 3, 4, 5, 6, 7, 8]:
        for k2 in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            if k1 >= k2:
                continue
            for a1 in [0.15, 0.25, 0.35]:
                for a2 in [0.1, 0.2, 0.3]:
                    h = 0.5 + a1*np.cos(2*np.pi*k1*np.arange(N)/N) + a2*np.cos(2*np.pi*k2*np.arange(N)/N)
                    h = np.clip(h, 0, 1)
                    h += (N/2.0 - np.sum(h)) / N
                    h = np.clip(h, 0, 1)
                    inits.append(h)
    
    # Random restarts
    for seed in range(40):
        np.random.seed(seed)
        h = np.random.uniform(0, 1, N)
        h += (N/2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        inits.append(h)
    
    # Perturbed sinusoidal starts
    np.random.seed(999)
    for k in [1, 2, 3, 4, 5, 6, 7, 8]:
        for noise_level in [0.05, 0.1, 0.2]:
            h = 0.5 + 0.45 * np.cos(2 * np.pi * k * np.arange(N) / N)
            h += noise_level * np.random.randn(N)
            h = np.clip(h, 0, 1)
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            inits.append(h)
    
    # Block-structured patterns
    for block_size in [2, 3, 4, 5, 10]:
        for offset in range(block_size):
            h = np.zeros(N)
            for i in range(N):
                block_idx = (i + offset) // block_size
                if block_idx % 2 == 0:
                    h[i] = 0.8
                else:
                    h[i] = 0.2
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            inits.append(h)
    
    # Triangular wave patterns
    for period in [4, 6, 8, 10, 12, 16, 20]:
        for amp in [0.3, 0.4, 0.45, 0.49]:
            t = np.arange(N)
            h = 0.5 + amp * (2 * np.abs(2 * (t / period - np.floor(t / period + 0.5))) - 1)
            h = np.clip(h, 0, 1)
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            inits.append(h)
    
    # Multi-pass refinement: use smooth objective with analytical gradient, then exact polish
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    for h_init in inits:
        if time.time() > deadline:
            break
        
        current_h = h_init.copy()
        
        # Multi-pass with increasing beta (increasing sharpness)
        for beta in beta_schedule:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    current_h,
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                if res.success:
                    current_h = res.x
            except Exception:
                pass
        
        # Final polish with exact objective
        if time.time() < deadline:
            try:
                res = minimize(
                    _objective,
                    current_h,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-11}
                )
                current_h = res.x
            except Exception:
                pass
        
        best_state = try_update(current_h, best_state)
    
    # Additional refinement passes on the best solution found
    if time.time() < deadline:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_state[0],
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                if res.success:
                    best_state = try_update(res.x, best_state)
            except Exception:
                pass
    
    # Final polish on best
    if time.time() < deadline:
        for ftol in [1e-12, 1e-14]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _objective,
                    best_state[0],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': ftol}
                )
                best_state = try_update(res.x, best_state)
            except Exception:
                pass
    
    best_h = best_state[0]
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        
    return best_h

# EVOLVE-BLOCK-END
