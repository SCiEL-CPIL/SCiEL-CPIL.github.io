"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
ALPHA = 500.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    s = np.sum(h)
    diff = N/2.0 - s
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max_and_grad(x, alpha=500.0):
    """Smooth approximation of max using log-sum-exp, returns value and gradient."""
    x_max = np.max(x)
    shifted = alpha * (x - x_max)
    # Clip to avoid overflow in exp, though x_max subtraction helps
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    val = x_max + (1.0/alpha) * np.log(sum_exp)
    grad = exp_shifted / sum_exp
    return val, grad

def _objective_and_grad(params, alpha=500.0):
    """Objective and gradient for L-BFGS-B.
    Objective: smooth_max(overlap) + penalty for sum constraint.
    """
    h = params
    # Compute overlap
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    
    smax, grad_smax = _smooth_max_and_grad(overlap, alpha)
    
    # Gradient of overlap wrt h
    # overlap[i] = h[i] * (1 - h[i+1])
    # d(overlap[i])/dh[i] = 1 - h[i+1]
    # d(overlap[i])/dh[i+1] = -h[i]
    
    grad_overlap = np.zeros(N)
    grad_overlap[:-1] += (1.0 - h[1:])  # contribution to h[i] from overlap[i]
    grad_overlap[1:] += (-h[:-1])       # contribution to h[i] from overlap[i-1]
    
    grad_total = grad_smax * grad_overlap
    
    # Penalty for sum constraint: lambda * (sum(h) - N/2)^2
    # We use a simple penalty term in the objective to guide towards feasibility
    # But L-BFGS-B doesn't support constraints directly well for equality.
    # Instead, we can use a barrier or just rely on the fact that the optimum
    # of the smooth max might naturally have sum close to N/2 if initialized well?
    # No, we must enforce sum.
    # Let's add a quadratic penalty to the objective.
    sum_h = np.sum(h)
    target = N/2.0
    penalty_val = 100.0 * (sum_h - target)**2
    penalty_grad = 200.0 * (sum_h - target) * np.ones(N)
    
    total_val = smax + penalty_val
    total_grad = grad_total + penalty_grad
    
    return total_val, total_grad

def _generate_initial_guess():
    """Generate a smooth bump initial guess."""
    # Create a sigmoid bump
    x = np.arange(N)
    center = N / 2.0
    width = N / 4.0
    # h = 1 / (1 + exp(-(x - center)/width)) - 1 / (1 + exp((x - center + 2*width)/width))
    # Simpler: use a cosine bump
    # h = 0.5 * (1 + cos(2*pi*x/N)) ? No, that's 0 at ends, 1 at middle.
    # We want sum = N/2.
    # Cosine bump: h_i = 0.5 * (1 + cos(2*pi*i/N)) has mean 0.5.
    # Max 1, Min 0.
    h = 0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / N))
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    def update(h, score):
        nonlocal best_h, best_score
        if h is not None and score < best_score:
            if np.all(h >= -1e-10) and np.all(h <= 1.0 + 1e-10) and abs(np.sum(h) - N/2.0) < 1e-6:
                best_h = h.copy()
                best_score = score
    
    # Phase 1: L-BFGS-B optimization from smooth bump
    h_init = _generate_initial_guess()
    h_init = _normalize(h_init)
    
    try:
        result = minimize(
            _objective_and_grad,
            h_init,
            method='L-BFGS-B',
            jac=True, # Returns (f, g)
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.success or result.status == 0:
            h_opt = _normalize(result.x)
            score_opt = _score(h_opt)
            update(h_opt, score_opt)
    except Exception:
        pass
    
    # Phase 2: Try a few different initializations if time permits
    if time.time() < deadline - 20.0:
        # Try a sharper bump
        h_init2 = np.zeros(N)
        h_init2[N//4 : 3*N//4] = 1.0
        # Smooth it
        h_init2 = np.convolve(h_init2, np.ones(10)/10, mode='same')
        h_init2 = _normalize(h_init2)
        
        try:
            result2 = minimize(
                _objective_and_grad,
                h_init2,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            if result2.success or result2.status == 0:
                h_opt2 = _normalize(result2.x)
                score_opt2 = _score(h_opt2)
                update(h_opt2, score_opt2)
        except Exception:
            pass

    # Phase 3: Polish with higher alpha if time permits
    if best_h is not None and time.time() < deadline - 10.0:
        # Re-optimize with higher alpha for sharper max approximation
        global ALPHA
        old_alpha = ALPHA
        ALPHA = 2000.0
        
        try:
            result3 = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            if result3.success or result3.status == 0:
                h_opt3 = _normalize(result3.x)
                score_opt3 = _score(h_opt3)
                update(h_opt3, score_opt3)
        except Exception:
            pass
        finally:
            ALPHA = old_alpha

    if best_h is None:
        # Fallback
        best_h = _generate_initial_guess()
        best_h = _normalize(best_h)
    
    return best_h
