import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 100
DEADLINE_OFFSET = 145.0

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(20):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _exact_objective(h):
    """Exact objective for global search."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _smooth_objective(h, lam):
    """
    Smoothed objective for local polish.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    shifted = corr - m
    val = m + np.log(np.sum(np.exp(lam * shifted))) / lam
    return val

def _slsqp_polish(h_start, time_budget=10.0):
    """
    Local refinement using SLSQP with increasing smoothness.
    """
    start_time = time.time()
    current_h = h_start.copy()
    best_h = h_start.copy()
    best_val = _correlation_value(best_h)
    
    lambdas = [10.0, 50.0, 200.0, 1000.0]
    
    for lam in lambdas:
        if time.time() - start_time > time_budget - 1.0:
            break
        
        def obj(x):
            return _smooth_objective(x, lam)
        
        try:
            res = minimize(
                obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
            current_h = h_opt
        except Exception:
            pass
            
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET
    
    # Initial best: uniform
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Global Search using Differential Evolution
    # DE is robust and does not require gradients.
    # We define bounds [0, 1] for each variable.
    # We need to handle the sum constraint. 
    # DE does not handle equality constraints natively well.
    # Strategy: Use DE to find a point close to the constraint, then project.
    # Or, use a penalty method in the objective for DE.
    
    def de_objective(x):
        # Project to feasible set to ensure we evaluate valid points?
        # No, DE explores the box. We should penalize constraint violation.
        # But projection is cheap. Let's project inside the objective to guide DE towards the manifold.
        h_proj = _project_to_feasible(x)
        val = _exact_objective(h_proj)
        return val
    
    # Run DE
    # Budget: 80 seconds
    de_time_budget = 80.0
    if time.time() < deadline - de_time_budget:
        try:
            # popsize=15, maxiter=150 should be fast enough with N=100
            # init='latinhypercube' for better coverage
            res_de = differential_evolution(
                de_objective,
                bounds=[(0.0, 1.0)] * N,
                popsize=15,
                maxiter=150,
                tol=1e-6,
                seed=42,
                init='latinhypercube',
                polish=False, # We will do our own polish
                updateinterval=0, # Disable printing
                disp=False
            )
            h_de = _project_to_feasible(res_de.x)
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
        except Exception:
            pass

    # Phase 2: Local Polish on the best found
    # Budget: remaining time minus safety margin
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 5.0:
        try:
            best_h = _slsqp_polish(best_h, time_budget=min(remaining_time, 30.0))
            best_val = _correlation_value(best_h)
        except Exception:
            pass
            
    return best_h
