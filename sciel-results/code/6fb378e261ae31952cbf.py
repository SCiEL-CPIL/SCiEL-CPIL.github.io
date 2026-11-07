"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2.0

def _smooth_max_corr(h, beta=50.0):
    """Smooth approximation of max correlation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    # Use log-sum-exp for smooth max
    m = np.max(corr)
    if m > 0:
        return m + np.log(np.sum(np.exp((corr - m) * beta)) / len(corr)) / beta
    else:
        return m + np.log(np.sum(np.exp(corr * beta)) / len(corr)) / beta

def _project_to_feasible(h):
    """Project h onto feasible set: sum=N/2, 0<=h<=1."""
    h = np.array(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(100):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _coordinate_descent(h, max_iters=500, tol=1e-10):
    """Coordinate descent to minimize max correlation."""
    h = np.array(h, dtype=float).copy()
    h = _project_to_feasible(h)
    best_val = _correlation_value(h)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Try to adjust h[i] while keeping sum constant
            # Adjust h[i] by delta, and compensate with another element
            # For simplicity, try small perturbations
            for delta in [0.01, -0.01, 0.05, -0.05]:
                h_new = h.copy()
                h_new[i] += delta
                if h_new[i] < 0 or h_new[i] > 1:
                    continue
                # Compensate by adjusting another element
                # Find the element with most room
                if delta > 0:
                    comp_mask = h_new < 1.0
                    comp_mask[i] = False
                else:
                    comp_mask = h_new > 0.0
                    comp_mask[i] = False
                if not np.any(comp_mask):
                    continue
                # Distribute compensation
                comp_idx = np.where(comp_mask)[0]
                comp_per = -delta / len(comp_idx)
                h_new[comp_idx] += comp_per
                h_new = np.clip(h_new, 0.0, 1.0)
                
                val = _correlation_value(h_new)
                if val < best_val - tol:
                    h = h_new.copy()
                    best_val = val
                    improved = True
                    break
        if not improved:
            break
    
    return h, best_val

def _generate_starts(n_starts=8, seed=42):
    """Generate diverse starting points."""
    rng = np.random.RandomState(seed)
    starts = []
    
    # Start 1: uniform 0.5
    starts.append(np.full(N, 0.5))
    
    # Start 2: alternating
    alt = np.ones(N)
    alt[::2] = 0.0
    alt[1::2] = 1.0
    # Adjust to sum = N/2
    alt = _project_to_feasible(alt)
    starts.append(alt)
    
    # Start 3-8: random
    for _ in range(n_starts - 2):
        h = rng.uniform(0.1, 0.9, size=N)
        h = _project_to_feasible(h)
        starts.append(h)
    
    return starts

def _slsqp_optimize(h_init, deadline):
    """Optimize using SLSQP with smooth objective."""
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    
    def obj(x):
        return _smooth_max_corr(x, beta=30.0)
    
    try:
        result = minimize(
            obj, h_init, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-10}
        )
        if time.time() > deadline:
            return h_init, _correlation_value(h_init)
        
        h_cand = _project_to_feasible(result.x)
        val = _correlation_value(h_cand)
        return h_cand, val
    except Exception:
        return h_init, _correlation_value(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    starts = _generate_starts(n_starts=8, seed=42)
    
    best_h = starts[0].copy()
    best_val = _correlation_value(best_h)
    
    for i, h_init in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        
        # SLSQP optimization
        h_opt, val_opt = _slsqp_optimize(h_init, deadline)
        
        # Coordinate descent polish
        if time.time() < deadline - 5.0:
            h_cd, val_cd = _coordinate_descent(h_opt, max_iters=200)
            if val_cd < val_opt:
                h_opt = h_cd
                val_opt = val_cd
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Final validation
    best_h = _project_to_feasible(best_h)
    return best_h
# EVOLVE-BLOCK-END
