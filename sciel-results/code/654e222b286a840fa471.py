"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float)
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        # Distribute correction uniformly, but be careful with clipping
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        # If clipping broke the sum, do a second pass on non-saturated elements
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                # Third pass if necessary
                diff3 = target - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max_and_grad(x, beta):
    """
    Compute soft max of x and its gradient using log-sum-exp.
    Returns (value, gradient_vector).
    """
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    
    # Shift for numerical stability
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    
    if wsum == 0:
        return max_x, np.zeros_like(x)
        
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad

def _objective_and_grad(h, beta):
    """
    Compute the smooth objective value and its gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    
    # Calculate overlap vector: o_i = h_i * (1 - h_{i+1})
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
        
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    
    # Gradient of overlap w.r.t h
    grad_h = np.zeros(n)
    if n > 1:
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0
        
    # Penalty term for sum constraint
    penalty_weight = 100.0
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad

def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """
    Optimize h using L-BFGS-B with analytic gradients and adaptive beta.
    """
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    # We will run a few stages with increasing beta
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]
    
    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break
            
        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad
        
        try:
            res = minimize(
                obj_func, 
                h, 
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception as e:
            pass
            
    return current_score, h

def _local_search_polish(h, max_iters=200, deadline=None):
    """
    Deterministic local search to polish the solution.
    Tries small adjustments and swaps.
    """
    h = h.copy()
    current_score = _score(h)
    
    if deadline is None:
        deadline = time.time() + 5.0
        
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try adjusting h[i] slightly
            for delta in [-0.01, -0.001, 0.001, 0.01]:
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = _fix_sum(h_trial)
                sc = _score(h_trial)
                if sc < current_score - 1e-12:
                    h = h_trial
                    current_score = sc
                    improved = True
                    break
        
        if not improved:
            break
            
    return current_score, h

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Initial guess: 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    h0 = _fix_sum(h0)
    current_score = _score(h0)
    h_final = h0.copy()

    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _fix_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - TARGET_SUM) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Direct optimization with analytic gradients
    try:
        if time.time() < deadline - 30.0:
            score, h_opt = _optimize_direct(h_final, deadline - 10.0)
            if score < current_score:
                _try_update(h_opt)
    except Exception:
        pass

    # Phase 2: Local search polish
    try:
        if time.time() < deadline - 10.0:
            score, h_polished = _local_search_polish(h_final, max_iters=100, deadline=deadline - 5.0)
            if score < current_score:
                _try_update(h_polished)
    except Exception:
        pass

    # Phase 3: Final tight polish with SLSQP
    try:
        if time.time() < deadline - 3.0:
            def objective_tight(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - TARGET_SUM}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_tight,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-16, 'disp': False}
            )

            if result.fun < current_score:
                _try_update(result.x)

    except Exception:
        pass

    # Ensure final result is valid
    h_final = _fix_sum(h_final)
    return h_final
