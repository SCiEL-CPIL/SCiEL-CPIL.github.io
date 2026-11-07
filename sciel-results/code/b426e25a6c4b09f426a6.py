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
    """Numerical gradient of the smooth objective."""
    eps = 1e-7
    n = len(h)
    grad = np.zeros(n)
    for i in range(n):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        grad[i] = (_smooth_objective(h_plus, beta) - _smooth_objective(h_minus, beta)) / (2 * eps)
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
    
    # Generate structured initializations
    inits = []
    
    # Cosine-based inits
    for k in [1, 2, 3, 4, 5]:
        for amp in [0.3, 0.4, 0.45, 0.49]:
            h = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            inits.append(h)
    
    # Multi-frequency
    for k1 in [1, 2, 3]:
        for k2 in [3, 4, 5, 7]:
            for a1 in [0.25, 0.3]:
                for a2 in [0.15, 0.2]:
                    h = 0.5 + a1*np.cos(2*np.pi*k1*np.arange(N)/N) + a2*np.cos(2*np.pi*k2*np.arange(N)/N)
                    h = np.clip(h, 0, 1)
                    h += (N/2.0 - np.sum(h)) / N
                    h = np.clip(h, 0, 1)
                    inits.append(h)
    
    # Random restarts
    np.random.seed(123)
    for _ in range(15):
        h = np.random.uniform(0, 1, N)
        h += (N/2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        inits.append(h)
    
    # Multi-pass refinement with decreasing beta for each init
    beta_schedule = [10.0, 30.0, 100.0]
    
    for h_init in inits:
        if time.time() > deadline:
            break
        
        current_h = h_init.copy()
        
        # Multi-pass with decreasing beta (increasing sharpness)
        for beta in beta_schedule:
            if time.time() > deadline:
                break
            
            # Use smooth objective for optimization
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    current_h,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10}
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
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                current_h = res.x
            except Exception:
                pass
        
        best_state = try_update(current_h, best_state)
    
    # Additional refinement passes on the best solution found
    if time.time() < deadline:
        for beta in [50.0, 200.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_state[0],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-11}
                )
                if res.success:
                    best_state = try_update(res.x, best_state)
            except Exception:
                pass
    
    # Final polish on best
    if time.time() < deadline:
        try:
            res = minimize(
                _objective,
                best_state[0],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14}
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
